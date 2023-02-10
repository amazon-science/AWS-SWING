import numpy as np
import rouge

from nltk import sent_tokenize
from transformers import BertForSequenceClassification, AutoTokenizer
import torch


def compute_rouge(preds, golds, return_all=False):


    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=False,
                           apply_avg=False,
                           apply_best=True,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
                           
    scores = evaluator.get_scores(preds, golds)
    avg_r = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
    # avg_r = scores['rouge-l']['f']

    print("Recall")
    print({'r1_recall': f"{scores['rouge-1']['r']*100:.2f}",
    'r2_recall': f"{scores['rouge-2']['r']*100:.2f}",
    'rL_recall': f"{scores['rouge-l']['r']*100:.2f}",
    })
    if return_all:
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=False,
                           apply_avg=False,
                           apply_best=False,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

        scores = evaluator.get_scores(preds, golds)
        return scores['rouge-l']
    else:
        return {'avg_r':avg_r, 
                'r1': scores['rouge-1']['f'],
                'r2': scores['rouge-2']['f'],
                'rL': scores['rouge-l']['f']}


def compute_bart_score(preds, golds, reverse=False, return_all=False):
    from bart_score import BARTScorer
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')

    if reverse:
        if isinstance(golds[0], list):
            bart_scores = bart_scorer.multi_ref_score(preds, golds, agg="max", batch_size=4)
        else:
            bart_scores = bart_scorer.score(preds, golds, batch_size=4)
    else:

        if isinstance(golds[0], list):
            bart_scores = []
            for idx, (pred, gold) in enumerate(zip(preds, golds)):
                this_bart_scores = []
                for g in gold:
                    this_bart_scores.append(bart_scorer.score([g], [pred], batch_size=1)[0])
                bart_scores.append(max(this_bart_scores))
            # bart_scores = bart_scorer.multi_ref_score(preds, golds, agg="max", batch_size=4)
        else:
            bart_scores = bart_scorer.score(golds, preds, batch_size=4)
    if return_all:
        return bart_scores
    else:
        return {'avg_r': np.mean(bart_scores)}



def compute_factcc_score_single_pair(factcc_model, factcc_tokenizer, document, sentence):
    factcc_model.eval()
    factcc_input_string = f"{document} [SEP] {sentence}"
    factcc_inputs = factcc_tokenizer(factcc_input_string, return_tensors='pt')
    sep_token_idx = torch.where(factcc_inputs['input_ids'][0] == factcc_tokenizer.sep_token_id)[0][0]

    factcc_inputs['token_type_ids'][0][sep_token_idx+1:] = 1
    for k, v in factcc_inputs.items():
        factcc_inputs[k] = v.to(factcc_model.device)

    outputs = factcc_model(**factcc_inputs, return_dict=True)
    logits = outputs.logits
    correct_prob = torch.nn.Softmax(dim=1)(logits)[0][0]
    
    return correct_prob.item()

@torch.no_grad()
def compute_factcc_score(preds, golds, reverse=False, return_all=False):
    
    # TODO: change this to the path you downloaded factCC to.
    factcc_model = model = BertForSequenceClassification.from_pretrained('/mnt/efs/dialogue_summ/factCC/models/factcc-checkpoint').cuda()
    factcc_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') 

    instances_correct_prob = []
    for idx, (pred, gold) in enumerate(zip(preds, golds)):

        # make everything multi-sentence
        if not isinstance(gold, list):
            gold = [gold]

        
        this_correct_prob = []
        if reverse:
            for g in gold:
                correct_probs = []
                gold_sents = sent_tokenize(g)
                for gold_sent in gold_sents:
                    correct_prob = compute_factcc_score_single_pair(factcc_model, factcc_tokenizer, document=pred, sentence = gold_sent)
                    correct_probs.append(correct_prob)
                if len(correct_probs) > 0:
                    this_correct_prob.append(np.mean(correct_probs))
                else:
                    this_correct_prob.append(0)
        else:
            pred_sents = sent_tokenize(pred)
            for g in gold:    
                correct_probs = []
                for pred_sent in pred_sents:
                    correct_prob = compute_factcc_score_single_pair(factcc_model, factcc_tokenizer, document=g, sentence = pred_sent)
                    correct_probs.append(correct_prob)
                if len(correct_probs) > 0:
                    this_correct_prob.append(np.mean(correct_probs) )
                else:
                    this_correct_prob.append(0)
        this_correct_prob = np.max(this_correct_prob)
        # this_correct_prob = np.mean(correct_probs)
        instances_correct_prob.append(this_correct_prob)
    if return_all:
        return [float(p) for p in instances_correct_prob]
    else:
        return np.mean(instances_correct_prob)

@torch.no_grad()
def compute_qafacteval_score(preds, golds, return_all=False):
    
    # we put the qafact eval import here
    from qafacteval import QAFactEval
    kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
            "verbose": True, "generation_batch_size": 8, \
            "answering_batch_size": 8, "lerc_batch_size": 2}
    
    # TODO: please change these paths to the corresponding path of the model you downloaded to.
    model_folder = '/mnt/efs/dialogue_summ/QAFactEval/models'
    metric = QAFactEval(
        lerc_quip_path=f"{model_folder}/quip-512-mocha",
        generation_model_path=f"{model_folder}/generation/model.tar.gz",
        answering_model_dir=f"{model_folder}/answering",
        lerc_model_path=f"{model_folder}/lerc/modnel.tar.gz",
        lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
        **kwargs
    )


    results = metric.score_batch(golds, [[sample] for sample in preds], return_qa_pairs=True)
    scores = [result[0]['qa-eval']['lerc_quip'] for result in results]
    if return_all:
        return scores
    else:
        
        return np.mean(scores)