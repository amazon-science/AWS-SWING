from xml.etree.ElementTree import TreeBuilder
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification
from typing import List, Dict
from types import MethodType
import numpy as np
import random
from collections import defaultdict
from utils import split_summary_into_sentences, label_smoothed_nll_loss
from transformers import BeamSearchScorer, LogitsProcessorList, TopKLogitsWarper

# define model
class FCBART(nn.Module):

    def __init__(self, model_name, args):
        super().__init__()
        
        self.bart = AutoModelForSeq2SeqLM.from_pretrained(model_name, gradient_checkpointing=args.do_gradient_checkpointing)
        

    def forward(self, 
                input_ids: torch.tensor, 
                attention_mask: torch.tensor, 
                decoder_input_ids: torch.tensor,
                decoder_labels: torch.tensor,
                decoder_label_strings=None,
                epoch=None
                ):

        outputs = self.bart(input_ids=input_ids,
                            attention_mask=attention_mask, #1 for tokens that are not masked, 0 for tokens that are masked.
                            decoder_input_ids=decoder_input_ids, # For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided, the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
                            labels=decoder_labels.reshape(-1), 
                            return_dict=True)
        
        loss = outputs['loss']

        return loss

    def generate(self, 
                input_ids: torch.tensor, 
                tokenizer):

        decoded_ids = self.bart.generate(inputs = input_ids, 
                        max_length=128, 
                        do_sample=True, 
                        num_beams=4, 
                        top_k=50, 
                        # no_repeat_ngram_size=3,
                        early_stopping=False,
                        use_cache=True)

        decoded_strings = tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
        
        res = {
            'decoded_ids': decoded_ids,
            'decoded_strings': decoded_strings
        }

        return res

class RobustBART(FCBART):
    '''
    Part of the code is from https://github.com/seanie12/CLAPS/blob/main/src/summarization/models.py
    '''
    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        self.tau = 0.1
        self.neg_eps = 1.0
        self.pos_eps = 3.0
        
    def forward(self,
                input_ids: torch.tensor, 
                attention_mask: torch.tensor, 
                decoder_input_ids: torch.tensor,
                decoder_labels: torch.tensor,
                decoder_label_strings: List[str]=None):

        encoder = self.bart.get_encoder()
        decoder = self.bart.get_decoder()

        encoder_outputs = encoder(input_ids=input_ids,
                                  attention_mask=attention_mask)

        hidden_states = encoder_outputs[0]

        decoder_attention_mask = torch.ones(decoder_input_ids.size()).to(hidden_states.device)
        decoder_attention_mask = decoder_attention_mask.masked_fill(decoder_input_ids == self.bart.config.pad_token_id, 0)

        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            # past_key_value_states=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=True,
        )
        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        # sequence_output = sequence_output * (self.bart.model_dim ** -0.5)
        lm_logits = self.bart.lm_head(sequence_output) + self.bart.final_logits_bias

        # Add hidden states and attention if they are here
        decoder_outputs = (lm_logits,) + decoder_outputs[1:]

        vocab_size = lm_logits.size(-1)

        criterion = nn.CrossEntropyLoss()
        nll = criterion(lm_logits.view(-1, vocab_size),
                        decoder_labels.reshape(-1))

        # adversarial
        proj_enc_h = hidden_states
        proj_dec_h = sequence_output
        avg_doc = self.avg_pool(proj_enc_h, attention_mask)

        
        
        # TODO: form decoder attention mask by computing the padding location in decoder_input_ids 
        avg_abs = self.avg_pool(proj_dec_h, decoder_attention_mask)

        cos = nn.CosineSimilarity(dim=-1)
        cont_crit = nn.CrossEntropyLoss()
        sim_matrix = cos(avg_doc.unsqueeze(1),
                            avg_abs.unsqueeze(0))
        perturbed_dec = self.generate_adv(sequence_output,
                                            decoder_labels)  # [n,b,t,d] or [b,t,d]
        batch_size = input_ids.size(0)

        proj_pert_dec_h = perturbed_dec
        avg_pert = self.avg_pool(proj_pert_dec_h,
                                    decoder_attention_mask)

        adv_sim = cos(avg_doc, avg_pert).unsqueeze(1)  # [b,1]

        pos_dec_hidden = self.generate_cont_adv(hidden_states, attention_mask,
                                                sequence_output, decoder_attention_mask,
                                                lm_logits,
                                                self.tau, self.pos_eps)
        avg_pos_dec = self.avg_pool(pos_dec_hidden,
                                    decoder_attention_mask)

        pos_sim = cos(avg_doc, avg_pos_dec).unsqueeze(-1)  # [b,1]
        logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau

        identity = torch.eye(batch_size, device=input_ids.device)
        pos_sim = identity * pos_sim
        neg_sim = sim_matrix.masked_fill(identity == 1, 0)
        new_sim_matrix = pos_sim + neg_sim
        new_logits = torch.cat([new_sim_matrix, adv_sim], 1)

        labels = torch.arange(batch_size,
                                device=input_ids.device)

        cont_loss = cont_crit(logits, labels)
        new_cont_loss = cont_crit(new_logits, labels)

        cont_loss = 0.5 * (cont_loss + new_cont_loss)

        return nll + cont_loss

    def generate_adv(self, dec_hiddens, lm_labels):
        dec_hiddens = dec_hiddens.detach()

        dec_hiddens.requires_grad = True

        lm_logits = self.bart.lm_head(dec_hiddens)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(lm_logits.view(-1, lm_logits.size(-1)),
                         lm_labels.reshape(-1))

        loss.backward()

        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturbed_dec = dec_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_dec = perturbed_dec  # [b,t,d]
        self.zero_grad()

        return perturbed_dec

    def generate_cont_adv(self, enc_hiddens, enc_mask,
                          dec_hiddens, dec_mask, lm_logits,
                          tau, eps):
        enc_hiddens = enc_hiddens.detach()
        dec_hiddens = dec_hiddens.detach()
        lm_logits = lm_logits.detach()
        dec_hiddens.requires_grad = True

        avg_enc = self.avg_pool(enc_hiddens,
                                enc_mask)

        avg_dec = self.avg_pool(dec_hiddens,
                                dec_mask)

        cos = nn.CosineSimilarity(dim=-1)
        logits = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_enc.size(0),
                              device=enc_hiddens.device)
        loss = cont_crit(logits, labels)
        loss.backward()

        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = dec_hiddens + eps * dec_grad
        perturb_dec_hidden = perturb_dec_hidden.detach()
        perturb_dec_hidden.requires_grad = True
        perturb_logits = self.bart.lm_head(perturb_dec_hidden)

        true_probs = F.softmax(lm_logits, -1)
        true_probs = true_probs * dec_mask.unsqueeze(-1).float()

        perturb_log_probs = F.log_softmax(perturb_logits, -1)

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = lm_logits.size(-1)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.sum(dec_mask).float()
        kl.backward()

        kl_grad = perturb_dec_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = perturb_dec_hidden - eps * kl_grad
        # self.zero_grad()

        return perturb_dec_hidden

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

class NLIBART(FCBART):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        if args.do_invalid or args.do_uncovered:
            self.nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
            self.nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
        if args.do_factcc_uncovered or args.do_factcc_validate:
            self.factcc_model = model = BertForSequenceClassification.from_pretrained('/mnt/efs/dialogue_summ/factCC/models/factcc-checkpoint')
            self.factcc_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        self.valid_sentence_threshold = 0.5
        self.args = args
        
    def forward(self, 
                input_ids: torch.tensor, 
                attention_mask: torch.tensor, 
                decoder_input_ids: torch.tensor,
                decoder_labels: torch.tensor,
                decoder_label_strings: List[str],
                epoch: int = None
                ):

        valid_sentence_threshold = self.valid_sentence_threshold
        invalid_loss = 0
        uncovered_loss = 0

        mle_outputs = self.bart(input_ids=input_ids,
                            attention_mask=attention_mask, #1 for tokens that are not masked, 0 for tokens that are masked.
                            decoder_input_ids=decoder_input_ids, # For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided, the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
                            labels=decoder_labels.reshape(-1), 
                            return_dict=True,
                            output_hidden_states=True)
        
        mle_loss = mle_outputs['loss']
        encoder_hidden_states = mle_outputs.encoder_last_hidden_state # (batch_size, sequence_length, hidden_size)
        dialog_embeddings = encoder_hidden_states[:,0,:] # (batch_size, hidden_states)
        ref_decoder_hidden_states = mle_outputs.decoder_hidden_states[-1] # (batch, seq, emb_dim)
        warmup_epoch = 0 if self.args.debug else 2
        # only do the following in later epochs 
        if epoch >= warmup_epoch:
            

            if self.args.do_invalid or self.args.do_uncovered or self.args.do_factcc_uncovered:
                # penalize sub sequence with low entailment score

                generated_outputs = self.generate_with_scores(input_ids, attention_mask, self.tokenizer)
                generated_strings = generated_outputs['decoded_strings']
                generated_scores = generated_outputs['decoded_sequences_scores']
                generated_ids = generated_outputs['decoded_ids']
                if self.args.do_invalid or self.args.do_uncovered:
                    nli_matrices = self.compute_nli_scores(generated_strings, decoder_label_strings)

            if self.args.do_factcc_uncovered:
                factcc_scores = self.compute_factcc_scores(generated_strings, decoder_label_strings)

                batch_decoder_input_output_ids = []

                # valid_sample_idxs = []
                mix_and_math_summaries = self.construct_mix_and_match_summary_factcc(generated_strings, decoder_label_strings, factcc_scores)
                valid_sample_idxs = mix_and_math_summaries['valid_sample_idxs']
                batch_decoder_input_output_ids = mix_and_math_summaries['batch_decoder_input_output_ids']
                
                
                batched_gen_idx_unmatched = mix_and_math_summaries['batched_gen_idx_unmatched']
                batched_ref_idx_unmatched = mix_and_math_summaries['batched_ref_idx_unmatched']
                
                
                if len(valid_sample_idxs) > 0:
                    batch_decoder_input_output_ids = torch.LongTensor(batch_decoder_input_output_ids).to(self.bart.device)

                    uncovered_decoder_input_ids = batch_decoder_input_output_ids[:,:-1]
                    uncovered_decoder_labels = batch_decoder_input_output_ids[:,1:]

                    uncovered_mle_outputs = self.bart(input_ids=input_ids[valid_sample_idxs],
                                    attention_mask=attention_mask[valid_sample_idxs], #1 for tokens that are not masked, 0 for tokens that are masked.
                                    decoder_input_ids=uncovered_decoder_input_ids, # For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided, the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
                                    labels=uncovered_decoder_labels.reshape(-1), 
                                    return_dict=True)
                    uncovered_loss = uncovered_mle_outputs['loss']

            # The uncovered loss
            if self.args.do_uncovered:

                # construct mix and match summaries
                mix_and_math_summaries = self.construct_mix_and_match_summary(generated_strings, decoder_label_strings, nli_matrices)
                valid_sample_idxs = mix_and_math_summaries['valid_sample_idxs']
                batch_decoder_input_output_ids = mix_and_math_summaries['batch_decoder_input_output_ids']
                batched_gen_idx_unmatched = mix_and_math_summaries['batched_gen_idx_unmatched']
                batched_ref_idx_unmatched = mix_and_math_summaries['batched_ref_idx_unmatched']
                batched_gen2ref_idx = mix_and_math_summaries['batched_gen2ref_idx']
                

                if len(valid_sample_idxs) > 0:
                    batch_decoder_input_output_ids = torch.LongTensor(batch_decoder_input_output_ids).to(self.bart.device)

                    uncovered_decoder_input_ids = batch_decoder_input_output_ids[:,:-1]
                    uncovered_decoder_labels = batch_decoder_input_output_ids[:,1:]

                    uncovered_mle_outputs = self.bart(input_ids=input_ids[valid_sample_idxs],
                                    attention_mask=attention_mask[valid_sample_idxs], #1 for tokens that are not masked, 0 for tokens that are masked.
                                    decoder_input_ids=uncovered_decoder_input_ids, # For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided, the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
                                    labels=uncovered_decoder_labels.reshape(-1), 
                                    return_dict=True)
                    uncovered_loss = uncovered_mle_outputs['loss']
                
            
            # do_invalid = do the contrastive loss
            if self.args.do_invalid:
                
                generated_decoder_input_ids = generated_ids[:,:-1]
                generated_decoder_labels = generated_ids[:,1:]
                # get scores
                generated_teacher_forcing_outputs = self.bart(input_ids=input_ids,
                                    attention_mask=attention_mask, #1 for tokens that are not masked, 0 for tokens that are masked.
                                    decoder_input_ids=generated_decoder_input_ids, # For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided, the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
                                    return_dict=True,
                                    output_hidden_states=True)
                generated_logits = generated_teacher_forcing_outputs.logits
                gen_decoder_hidden_states = generated_teacher_forcing_outputs.decoder_hidden_states[-1] # (batch, seq, emb_dim)
                seq_generated_scores = F.softmax(generated_logits, dim=-1)
                generated_scores = torch.gather(seq_generated_scores, 2, generated_decoder_labels.unsqueeze(2)).squeeze(2)
                invalid_token_length = 0

                for this_dialog_embeddings, nli_matrix, this_decoder_label_ids, decoder_label_string, this_ref_decoder_hidden_states\
                , this_generated_ids, generated_string, this_generated_scores, this_gen_decoder_hidden_states, gen_idx_unmatched, gen2ref_idx in \
                    zip(dialog_embeddings, nli_matrices, decoder_labels, decoder_label_strings, ref_decoder_hidden_states\
                    , generated_ids, generated_strings, generated_scores, gen_decoder_hidden_states, batched_gen_idx_unmatched, batched_gen2ref_idx):
                    
                    
                    # if no sentences generated continue
                    if nli_matrix.size == 0:
                        continue
                    this_ref_embeddings = this_ref_decoder_hidden_states[0,:] # (emb)
                    
                    # the max entailment prob for each generated sentence
                    generated_best_entailment_probs = nli_matrix.max(axis=0) # (n_generated_sentences)

                    # only consider stuff appearing before the first eos token and after the first
                    
                    # if there are two eos
                    if (this_generated_ids == self.tokenizer.eos_token_id).sum() >= 2:
                        gen_eos_idx = torch.where(this_generated_ids == self.tokenizer.eos_token_id)[0][1]
                    else:
                        gen_eos_idx = len(this_generated_ids)
                    
                    

                    this_generated_ids = this_generated_ids[1:gen_eos_idx+1]
                    

                    if (this_decoder_label_ids == self.tokenizer.eos_token_id).sum() >= 1:
                        ref_eos_idx = torch.where(this_decoder_label_ids == self.tokenizer.eos_token_id)[0][0]
                    else:
                        ref_eos_idx = len(this_decoder_label_ids)

                    this_decoder_label_ids = this_decoder_label_ids[:ref_eos_idx+1]
                    
                    

                    # Idenify the index of the periods 
                    tokens = self.tokenizer.convert_ids_to_tokens(this_generated_ids)
                    period_indices = [idx for idx, tok in enumerate(tokens) if '.' in tok]
                    gen_segment_indices = [0] + period_indices + [len(this_generated_ids)]
                    
                    
                    
                    # Contrastive loss
                    for pos_gen_idx, ref_idx in gen2ref_idx.items():
                        
                        # only do contrastive if there are negative samples
                        if len(gen_idx_unmatched) > 0:

                            # get embeddings of all negative sentences in the current sample (summary)
                            this_all_neg_gen_sentence_embeddings = []
                            for neg_gen_idx in gen_idx_unmatched:
                                neg_gen_segment_start_idx, neg_gen_segment_end_idx = gen_segment_indices[neg_gen_idx]+1, gen_segment_indices[neg_gen_idx+1]+1
                                this_neg_gen_sentence_embeddings = this_gen_decoder_hidden_states[neg_gen_segment_start_idx: neg_gen_segment_end_idx,:].mean(dim=0)
                                this_all_neg_gen_sentence_embeddings.append(this_neg_gen_sentence_embeddings)
                            
                            # shape: (n_neg_sents, emb_dim)
                            this_all_neg_gen_sentence_embeddings = torch.stack(this_all_neg_gen_sentence_embeddings, dim=0) # (n_neg, dim)

                            # get the positive embedding start and end index
                            pos_gen_segment_start_idx, pos_gen_segment_end_idx = gen_segment_indices[pos_gen_idx]+1, gen_segment_indices[pos_gen_idx+1]+1
                
                            # form representations of the positive sentence
                            this_pos_gen_sentence_embeddings = this_gen_decoder_hidden_states[pos_gen_segment_start_idx: pos_gen_segment_end_idx,:].mean(dim=0)
                            
                            # compute cosine similarity
                            pos_logits = F.cosine_similarity(this_pos_gen_sentence_embeddings, this_dialog_embeddings, dim=0)
                            all_neg_logits = F.cosine_similarity(this_all_neg_gen_sentence_embeddings, this_dialog_embeddings.unsqueeze(0), dim=1)
                            
                            # concat all logits
                            logits = torch.cat([pos_logits.unsqueeze(dim=0), all_neg_logits], dim=-1)
                            
                            # compute cross entropy aka negative log likelihood
                            lprobs = F.log_softmax(logits, dim=-1)
                            positive_lprob = lprobs[0]
                            invalid_loss += - positive_lprob
                    
                    
            loss = mle_loss + self.args.invalid_weights * invalid_loss + self.args.uncovered_weights * uncovered_loss
            
        else:
            loss = mle_loss

        assert torch.isfinite(loss), (mle_loss, invalid_loss, uncovered_loss)
        return loss

        

    def compute_nli_scores(self, 
                        decoded_strings: List[str], 
                        decoder_label_strings: List[str]) -> List[np.array]:
        '''
        Output a list of matrix. Each matrix size is n_label_sentences * n_pred_sentences.
        '''
        with torch.no_grad():
            self.nli_model.eval()
            nli_matrices = []
            for decoder_label_string, decoded_string in zip(decoder_label_strings, decoded_strings):
                label_sentences = split_summary_into_sentences(decoder_label_string)
                decoded_sentences = split_summary_into_sentences(decoded_string)
                # print(len(label_sentences), len(decoded_sentences))

                # TODO: gather input_to_nli and get nli score in one pass
                nli_scores = np.zeros([len(label_sentences), len(decoded_sentences)])
                input_to_nlis = []
                
                for label_idx, label_sentence in enumerate(label_sentences):
                    for decoded_idx, decoded_sentence in enumerate(decoded_sentences):
                        
                        # remember to add period bc we remove period in `split('.')`.                
                        input_to_nli = f'{label_sentence} </s></s> {decoded_sentence}'
                        input_to_nlis.append(input_to_nli)

                        
                        
                
                if len(input_to_nlis) == 0:
                    nli_scores = np.array([[]])
                else:

                    # compute forward entailment prob from reference to generated
                    input_to_nlis = self.nli_tokenizer(input_to_nlis, truncation=True, max_length=128, padding='longest', return_tensors='pt')
                    n_nli_inputs = len(label_sentences) * len(decoded_sentences)
                    
                    nli_batch_size = 4
                    n_nli_batches = n_nli_inputs // nli_batch_size + 1

                    nli_scores = []
                    for nli_batch_idx in range(n_nli_batches):
                        batch_input_to_nlis = {}
                        for k, v in input_to_nlis.items():
                            batch_input_to_nlis[k] = v[nli_batch_idx * nli_batch_size: (nli_batch_idx+1) * nli_batch_size].to(self.nli_model.device)
                        nli_logits = self.nli_model(**batch_input_to_nlis).logits
                        
                        entailment_prob = F.softmax(nli_logits, dim=1)[:,2].cpu().numpy()
                        

                        nli_scores.append(entailment_prob)
                    
                    nli_scores = np.concatenate(nli_scores, axis=-1).reshape(len(label_sentences), len(decoded_sentences))

                    # compute backward entailment score from generated to reference
                    # backward_input_to_nlis = self.nli_tokenizer(backward_input_to_nlis, truncation=True, max_length=128, padding='longest', return_tensors='pt')
                    # n_nli_inputs = len(label_sentences) * len(decoded_sentences)
                    
                    # nli_batch_size = 4
                    # n_nli_batches = n_nli_inputs // nli_batch_size + 1

                    # backward_nli_scores = []
                    # for nli_batch_idx in range(n_nli_batches):
                    #     batch_backward_input_to_nlis = {}
                    #     for k, v in backward_input_to_nlis.items():
                    #         batch_backward_input_to_nlis[k] = v[nli_batch_idx * nli_batch_size: (nli_batch_idx+1) * nli_batch_size].to(self.nli_model.device)
                    #     nli_logits = self.nli_model(**batch_backward_input_to_nlis).logits
                        
                    #     entailment_prob = F.softmax(nli_logits, dim=1)[:,2].cpu().numpy()

                    #     backward_nli_scores.append(entailment_prob)
                    
                    
                    # backward_nli_scores = np.concatenate(backward_nli_scores, axis=-1).reshape(len(label_sentences), len(decoded_sentences))

                    # # we only care to adjust 
                    # adjust_mask = nli_scores > valid_sentence_threshold
                    # # fill this because 
                    # backward_nli_scores[nli_scores < valid_sentence_threshold] = 1
                    # # take the min among
                    # nli_scores = 
                nli_matrices.append(nli_scores)

        # nli_matrices = np.stack(nli_matrices, axis=0)

        # TODO: compute a backward entailable score for each sentence in the generated
        return nli_matrices
        
    @torch.no_grad()
    def compute_factcc_score_single_pair(self, document, sentence):
        self.factcc_model.eval()
        factcc_input_string = f"{document} [SEP] {sentence}"
        factcc_inputs = self.factcc_tokenizer(factcc_input_string, return_tensors='pt')
        sep_token_idx = torch.where(factcc_inputs['input_ids'][0] == self.factcc_tokenizer.sep_token_id)[0][0]

        factcc_inputs['token_type_ids'][0][sep_token_idx+1:] = 1
        for k, v in factcc_inputs.items():
            factcc_inputs[k] = v.to(self.factcc_model.device)

        outputs = self.factcc_model(**factcc_inputs, return_dict=True)
        logits = outputs.logits
        correct_prob = torch.nn.Softmax(dim=1)(logits)[0][0]
        
        return correct_prob

    @torch.no_grad()
    def construct_mix_and_match_summary(self,
                        decoded_strings: List[str], 
                        decoder_label_strings: List[str],
                        nli_matrices: List[np.array]) -> Dict:

        
        # nli_matrices = self.compute_nli_scores(decoded_strings, decoder_label_strings)
        batch_decoder_input_output_ids = []
        valid_sample_idxs = [] # some samples are invalid because the model outputs emtpy string
        batched_gen_idx_unmatched = [] 
        batched_ref_idx_unmatched = []
        batched_gen2ref_idx = []
        for sample_idx, (nli_matrix, decoded_string, decoder_label_string) in enumerate(zip(nli_matrices, decoded_strings, decoder_label_strings)):
            if nli_matrix.size == 0:
                continue
            # convert to np array bc we need list indexing
            label_sentences = np.array(split_summary_into_sentences(decoder_label_string))
            decoded_sentences = np.array(split_summary_into_sentences(decoded_string))
            # mapping from 
            valid_mapping  = defaultdict(list)

            # (x-indices, y-indices), such as (array([0, 2, 2]), array([2, 0, 2]))
            x_y_indices = np.where(nli_matrix > self.valid_sentence_threshold)

            ref2gen = defaultdict(list)
            gen2ref = defaultdict(list)
            for ref_idx, gen_idx in zip(*x_y_indices):
                ref2gen[ref_idx].append(gen_idx)
                gen2ref[gen_idx].append(ref_idx)

            
            
            # a set that records the idx of the sentence that can be matched in the reference summary 
            ref_idx_matched = set()
            gen_idx_matched = set()

            # gen2ref idx if both direction matched
            matched_gen2ref_idx = {}

            # Check 1:M (1 ref to multiple gens)
            for ref_idx, gen_indices in ref2gen.items():

                # multiple gens
                if len(gen_indices) >= 2:
                    gen_indices = sorted(gen_indices)
                    # all gens are consecutive
                    if gen_indices[-1] - gen_indices[0] + 1== len(gen_indices):
                        concat_decoded_sentences = ' '.join(decoded_sentences[gen_indices])
                        label_sentence = label_sentences[ref_idx]
                        backward_input_to_nli = f'{concat_decoded_sentences} </s></s> {label_sentence}'
                        
                        input_to_nlis = self.nli_tokenizer(backward_input_to_nli, truncation=True, max_length=128, padding='longest', return_tensors='pt')
                        
                        for k, v in input_to_nlis.items():
                            input_to_nlis[k] = v.to(self.nli_model.device)
                        nli_logits = self.nli_model(**input_to_nlis).logits
                        
                        entailment_prob = F.softmax(nli_logits, dim=1)[0,2].cpu().numpy()


                        factcc_correct_prob = 1
                        # also check factcc if enable factcc_validate and only check this if nli entailment prob is high enough to save computation
                        if self.args.do_factcc_validate and entailment_prob >= self.valid_sentence_threshold:
                            factcc_correct_prob = self.compute_factcc_score_single_pair(document=concat_decoded_sentences, sentence=label_sentence)

                        if entailment_prob >= self.valid_sentence_threshold and factcc_correct_prob >= self.valid_sentence_threshold:
                            ref_idx_matched.add(ref_idx)
                            gen_idx_matched = gen_idx_matched.union(gen_indices)
                            for gen_idx in gen_indices:
                                matched_gen2ref_idx[gen_idx] = ref_idx

            # Check M:1 (multiple ref to 1 gen) 
            for gen_idx, ref_indices in gen2ref.items():
                # make sure all ref idx has not been matched
                if len(ref_indices) >= 2 and all([ref_idx not in ref_idx_matched for ref_idx in ref_indices]) and gen_idx not in gen_idx_matched:
                    ref_indices = sorted(ref_indices)
                    if ref_indices[-1] - ref_indices[0] +1 == len(ref_indices):
                        concat_label_sentences = ' '.join(label_sentences[ref_indices])
                        decoded_sentence = decoded_sentences[gen_idx]
                        backward_input_to_nli = f'{decoded_sentence} </s></s> {concat_label_sentences}'
                        
                        input_to_nlis = self.nli_tokenizer(backward_input_to_nli, truncation=True, max_length=128, padding='longest', return_tensors='pt')

                        for k, v in input_to_nlis.items():
                            input_to_nlis[k] = v.to(self.nli_model.device)
                        nli_logits = self.nli_model(**input_to_nlis).logits
                        
                        entailment_prob = F.softmax(nli_logits, dim=1)[0,2].cpu().numpy()

                        factcc_correct_prob = 1
                        # also check factcc if enable factcc_validate and only check this if nli entailment prob is high enough to save computation
                        if self.args.do_factcc_validate and entailment_prob >= self.valid_sentence_threshold:
                            factcc_correct_prob = self.compute_factcc_score_single_pair(document=decoded_sentence, sentence=concat_label_sentences)

                        if entailment_prob >= self.valid_sentence_threshold and factcc_correct_prob >= self.valid_sentence_threshold:
                            ref_idx_matched = ref_idx_matched.union(ref_indices)
                            gen_idx_matched.add(gen_idx)

                            for ref_idx in ref_indices:
                                matched_gen2ref_idx[gen_idx] = ref_idx
            # Check 1:1
            gen2ref_idx = nli_matrix.argmax(axis=0) # find the corresponding ref sentence for each gen sentence

            for gen_idx, ref_idx in enumerate(gen2ref_idx):
                # only do this if this is greater than threshold and 
                if gen_idx not in gen_idx_matched and \
                ref_idx not in ref_idx_matched and \
                nli_matrix[ref_idx, gen_idx] > self.valid_sentence_threshold:


                    label_sentence = label_sentences[ref_idx]
                    decoded_sentence = decoded_sentences[gen_idx]
                    backward_input_to_nli = f'{decoded_sentence} </s></s> {label_sentence}'
                    
                    input_to_nlis = self.nli_tokenizer(backward_input_to_nli, truncation=True, max_length=128, padding='longest', return_tensors='pt')

                    for k, v in input_to_nlis.items():
                        input_to_nlis[k] = v.to(self.nli_model.device)
                    nli_logits = self.nli_model(**input_to_nlis).logits
                    
                    entailment_prob = F.softmax(nli_logits, dim=1)[0,2].cpu().numpy()
                    factcc_correct_prob = 1
                    # also check factcc if enable factcc_validate and only check this if nli entailment prob is high enough to save computation
                    if self.args.do_factcc_validate and entailment_prob >= self.valid_sentence_threshold:
                        factcc_correct_prob = self.compute_factcc_score_single_pair(document=decoded_sentence, sentence=label_sentence)

                    if entailment_prob >= self.valid_sentence_threshold and factcc_correct_prob >= self.valid_sentence_threshold:
                        ref_idx_matched.add(ref_idx)
                        gen_idx_matched.add(gen_idx)
                        matched_gen2ref_idx[gen_idx] = ref_idx

            ref_idx_sequences = [(ref_idx, ref_sent) for ref_idx, ref_sent in enumerate((label_sentences)) if ref_idx not in ref_idx_matched]
            gen_idx_sequences = [(matched_gen2ref_idx[gen_idx], gen_sent) for gen_idx, gen_sent in enumerate((decoded_sentences)) if gen_idx in gen_idx_matched]
            
            batched_gen_idx_unmatched.append([gen_idx for gen_idx in range(len(decoded_sentences)) if gen_idx not in gen_idx_matched])
            batched_ref_idx_unmatched.append([ref_idx for ref_idx in range(len(label_sentences)) if ref_idx not in ref_idx_matched])
            batched_gen2ref_idx.append(matched_gen2ref_idx)
            # only consider those that has at least one valid generated sentence
            if len(gen_idx_sequences) == 0: 
                continue
            mix_n_match_idx_sequences = ref_idx_sequences + gen_idx_sequences
            
            mix_n_match_sequences = [sent for idx, sent in sorted(mix_n_match_idx_sequences, key=lambda x:x[0])]

            mix_n_match_string = ' '.join(mix_n_match_sequences)
            
            decoder_input_output_ids = self.tokenizer.encode(mix_n_match_string, max_length=128, padding="max_length", truncation=True)
            if 'bart' in self.args.model_name:
                decoder_input_output_ids = [self.tokenizer.eos_token_id] + decoder_input_output_ids
            
            batch_decoder_input_output_ids.append(decoder_input_output_ids)
            valid_sample_idxs.append(sample_idx)
            
        return {
            'batch_decoder_input_output_ids':batch_decoder_input_output_ids,
            'valid_sample_idxs': valid_sample_idxs,
            'batched_gen_idx_unmatched': batched_gen_idx_unmatched,
            'batched_ref_idx_unmatched': batched_ref_idx_unmatched,
            'batched_gen2ref_idx': batched_gen2ref_idx
            
        }

    @torch.no_grad()
    def compute_factcc_scores(self, 
                        decoded_strings: List[str], 
                        decoder_label_strings: List[str]) -> List[Dict]:
        '''
        Output a list of matrix. Each matrix size is n_label_sentences * n_pred_sentences.
        '''
        
        self.factcc_model.eval()
        
        factcc_scores = {'generated_correct_probs':[], 'reference_correct_probs':[]}

        for decoder_label_string, decoded_string in zip(decoder_label_strings, decoded_strings):
            label_sentences = split_summary_into_sentences(decoder_label_string)
            decoded_sentences = split_summary_into_sentences(decoded_string)
            # print(len(label_sentences), len(decoded_sentences))

            # do not continue if either the generated or reference summary is empty.
            if len(decoded_sentences) == 0 or len(label_sentences) == 0: 
                factcc_scores['generated_correct_probs'].append([]) 
                factcc_scores['reference_correct_probs'].append([])
                continue
            
            this_generated_correct_probs = []
            this_reference_correct_probs = []
            input_to_factccs = []
            
            
            for decoded_idx, decoded_sentence in enumerate(decoded_sentences):
                # we put scores for each decoded sentence first
                factcc_input_string = f"{decoder_label_string} [SEP] {decoded_sentence}"
                input_to_factccs.append(factcc_input_string)
            
            for label_idx, label_sentence in enumerate(label_sentences):
                factcc_input_string = f"{decoded_string} [SEP] {label_sentence}"
                input_to_factccs.append(factcc_input_string)
                    
            
            

            # compute forward entailment prob from reference to generated
            input_to_factccs = self.factcc_tokenizer(input_to_factccs, truncation=True, max_length=256, padding='longest', return_tensors='pt')
            n_factcc_inputs = len(input_to_factccs) 
            
            factcc_batch_size = 4
            n_factcc_batches = n_factcc_inputs // factcc_batch_size + 1

            
            for factcc_batch_idx in range(n_factcc_batches):
                batch_input_to_factccs = {}
                for k, v in input_to_factccs.items():
                    batch_input_to_factccs[k] = v[factcc_batch_idx * factcc_batch_size: (factcc_batch_idx+1) * factcc_batch_size].to(self.factcc_model.device)

                # set token_type_ids correctly
                for sample_idx in range(min(factcc_batch_size, len(batch_input_to_factccs['input_ids']))):
                    sep_token_idx = torch.where(batch_input_to_factccs['input_ids'][sample_idx] == self.factcc_tokenizer.sep_token_id)[0][0]
                    batch_input_to_factccs['token_type_ids'][sample_idx][sep_token_idx+1:] = 1

                factcc_logits = self.factcc_model(**batch_input_to_factccs).logits
                
                correct_prob = F.softmax(factcc_logits, dim=1)[:,0].cpu().numpy()
                
                generated_correct_probs = correct_prob[:len(decoded_sentences)]
                reference_correct_probs = correct_prob[len(decoded_sentences):]
                
                this_generated_correct_probs.append(generated_correct_probs)
                this_reference_correct_probs.append(reference_correct_probs)

                
            
            this_generated_correct_probs = np.concatenate(this_generated_correct_probs, axis=-1)
            this_reference_correct_probs = np.concatenate(this_reference_correct_probs, axis=-1)

            factcc_scores['generated_correct_probs'].append(this_generated_correct_probs) 
            factcc_scores['reference_correct_probs'].append(this_reference_correct_probs)
            
            

        # TODO: compute a backward entailable score for each sentence in the generated
        return factcc_scores

    @torch.no_grad()
    def construct_mix_and_match_summary_factcc(self,
                        decoded_strings: List[str], 
                        decoder_label_strings: List[str],
                        factcc_scores: List[Dict]) -> Dict:
        
        
        
        batch_decoder_input_output_ids = []
        valid_sample_idxs = [] # some samples are invalid because the model outputs emtpy string
        batched_gen_idx_unmatched = [] 
        batched_ref_idx_unmatched = []
        
        fact_cc_correct_probs = []
        generated_correct_probs = factcc_scores['generated_correct_probs']
        reference_correct_probs = factcc_scores['reference_correct_probs']

        # make sure the sample size is the same
        assert len(generated_correct_probs) == len(reference_correct_probs) == len(decoded_strings)

        
        for sample_idx, (this_generated_correct_probs, this_reference_correct_probs, decoded_string, decoder_label_string) in enumerate(zip(generated_correct_probs, reference_correct_probs, decoded_strings, decoder_label_strings)):
            if len(this_generated_correct_probs) == 0 or len(this_reference_correct_probs) == 0: continue
            
            # convert to np array bc we need list indexing
            label_sentences = np.array(split_summary_into_sentences(decoder_label_string))
            decoded_sentences = np.array(split_summary_into_sentences(decoded_string))

            ref_idx_matched = set([idx for idx, prob in enumerate(this_reference_correct_probs) if prob > 0.5])
            gen_idx_matched = set([idx for idx, prob in enumerate(this_generated_correct_probs) if prob > 0.5])

            ref_idx_sequences = [(ref_idx, ref_sent) for ref_idx, ref_sent in enumerate((label_sentences)) if ref_idx not in ref_idx_matched]

            gen_idx_sequences = [(gen_idx, gen_sent) for gen_idx, gen_sent in enumerate((decoded_sentences)) if gen_idx in gen_idx_matched]
            
            batched_gen_idx_unmatched.append([gen_idx for gen_idx in range(len(decoded_sentences)) if gen_idx not in gen_idx_matched])
            batched_ref_idx_unmatched.append([ref_idx for ref_idx in range(len(label_sentences)) if ref_idx not in ref_idx_matched])
            # only consider those that has at least one valid generated sentence
            if len(gen_idx_sequences) == 0: 
                continue
            mix_n_match_idx_sequences = ref_idx_sequences + gen_idx_sequences
            
            mix_n_match_sequences = [sent for idx, sent in sorted(mix_n_match_idx_sequences, key=lambda x:x[0])]

            mix_n_match_string = ' '.join(mix_n_match_sequences)
            
            decoder_input_output_ids = self.tokenizer.encode(mix_n_match_string, max_length=128, padding="max_length", truncation=True)
            if 'bart' in self.args.model_name:
                decoder_input_output_ids = [self.tokenizer.eos_token_id] + decoder_input_output_ids
            
            batch_decoder_input_output_ids.append(decoder_input_output_ids)
            valid_sample_idxs.append(sample_idx)
            
        return {
            'batch_decoder_input_output_ids':batch_decoder_input_output_ids,
            'valid_sample_idxs': valid_sample_idxs,
            'batched_gen_idx_unmatched': batched_gen_idx_unmatched,
            'batched_ref_idx_unmatched': batched_ref_idx_unmatched
            
        }

    def construct_positive_samples(self, 
                        decoded_strings: List[str], 
                        decoder_label_strings: List[str],
                        nli_matrices: np.array):
        label_sentences = decoder_label_strings.split('.')
        decoded_sentences = decoded_strings.split('.')

        positive_samples = []

        for decoder_label_string, decoded_string, nli_matrix in zip(decoder_label_strings, decoded_strings, nli_matrices):
            label_sentences = decoder_label_string.split('.')
            decoded_sentences = decoded_string.split('.')
            valid_sentence_idxs = np.where(nli_matrix.max(axis=0) > 0.5)

            matching_idxs = nli_matrix.argmax(axis=0)

            # TODO: identify the matched indices

    def generate_with_grad(self, 
                input_ids: torch.tensor, 
                attention_mask: torch.tensor):

        num_beams = 4
        decoding_length = 128
        beam_scorer = BeamSearchScorer(
            batch_size=input_ids.size(0),
            # max_length=decoding_length,
            num_beams=num_beams,
            device=self.bart.device,
        )

        logits_processor = LogitsProcessorList([])

        logits_warper = LogitsProcessorList([TopKLogitsWarper(top_k=50)])
        
        # seems that this is required if our model is a encoder-decoder architecture.
        model_kwargs = {
            "encoder_outputs": self.bart.get_encoder()(input_ids.repeat_interleave(num_beams, dim=0), attention_mask.repeat_interleave(num_beams, dim=0), return_dict=False),
        }
        

        # create token for start decoding.
        decoder_input_ids = torch.ones((num_beams * input_ids.size(0), 1), device=self.bart.device, dtype=torch.long)
        decoder_input_ids = decoder_input_ids * self.bart.config.decoder_start_token_id
        
         
        
        return self.bart.beam_sample(decoder_input_ids, beam_scorer, max_length=decoding_length, logits_processor=logits_processor, logits_warper=logits_warper, return_dict_in_generate=True, output_scores=True, **model_kwargs)


    def generate_with_scores(self,
                input_ids: torch.tensor, 
                attention_mask: torch.tensor,
                tokenizer) -> Dict:
        beam_size = 4

        decoded_outputs = self.bart.generate(inputs = input_ids, 
                    max_length=128, 
                    do_sample=True, 
                    num_beams=beam_size, 
                    top_k=50, 
                    # no_repeat_ngram_size=3,
                    early_stopping=False,
                    use_cache=True,
                    output_scores=True,
                    return_dict_in_generate=True)
    
        decoded_ids = decoded_outputs.sequences
        decoded_sequences_scores = None

        decoded_strings = tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
        
        res = {
            'decoded_ids': decoded_ids,
            'decoded_strings': decoded_strings,
            'decoded_sequences_scores': decoded_sequences_scores,
        }

        return res