from transformers import Adafactor, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import random
import time
import os
from model import FCBART, RobustBART, NLIBART
from evals import compute_rouge, compute_bart_score
from args import create_args
from data import SummDataset
from constants import DIALOGSUM, SAMSUM


seed = 42
# fix random seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False

# use args as configuration
args = create_args()

# make sure the output_dir does not exist, 
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


output_dir = os.path.join(args.output_dir, args.exp_name)
os.makedirs(output_dir)
# init model and tokenizer

# Save configuration
args_dict = vars(args)
with open(os.path.join(output_dir, 'config.json'),'w') as f:
    json.dump(args_dict, f)

if args.use_robust:
    # the model that do adversarial training tried in the early stage of the experiments.
    model = RobustBART(args.model_name, args).cuda()
elif args.use_nli:
    # the proposed model 
    model = NLIBART(args.model_name, args).cuda()
else:
    # FCBART is just BART
    model = FCBART(args.model_name, args).cuda()
assert not (args.use_robust and args.use_nli), "only one of these argument can be true: use_robust, use_nli"



# Load data
if args.dataset == SAMSUM:
    args.data_dir = '../data/samsum'
elif args.dataset == DIALOGSUM:
    args.data_dir = '../data/dialogsum/DialogSum_Data'
else:
    raise NotImplementedError
print(f'Reading data from {args.data_dir}')

train_path = os.path.join(args.data_dir, 'train.json')
dev_path = os.path.join(args.data_dir, 'val.json')
test_path = os.path.join(args.data_dir, 'test.json')

# Init data loader
train_set = SummDataset(train_path, args)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
dev_set = SummDataset(dev_path, args)
dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn)
test_set = SummDataset(test_path, args, is_test=True)
test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn)


state = dict(model=model.state_dict())



batch_num = len(train_set) // (args.batch_size * args.accumulate_step)
+ (len(train_set) % (args.batch_size * args.accumulate_step) != 0)

# optimizer
if 't5' in args.model_name:
    optimizer = Adafactor(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, scale_parameter=False, relative_step=False, warmup_init=False)
else:
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num*args.warmup_epoch,
                                           num_training_steps=batch_num*args.max_epoch)

best_dev_f1 = -np.inf

model_path = os.path.join(output_dir,'best.pt')
for epoch in range(args.max_epoch):
    training_loss = 0
    model.train()
    for batch_idx, (input_ids, attn_mask, decoder_input_output_ids, target_strings) in enumerate(tqdm(train_loader)):        
        
        decoder_input_ids = decoder_input_output_ids[:,:-1]
        decoder_labels = decoder_input_output_ids[:,1:]

        loss = model(input_ids, 
                attention_mask=attn_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_labels=decoder_labels,
                decoder_label_strings=target_strings,
                epoch=epoch)
        
        loss.backward()
        training_loss += loss.item()
        if (batch_idx + 1) % args.accumulate_step == 0:

            # don't do gradient clipping when using adafactor
            if 't5' not in args.model_name:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0)
        
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        if args.debug and batch_idx >= 200:
            break
    print("Training loss", training_loss)

    # Train the last batch
    if batch_num % args.accumulate_step != 0:
        if 't5' not in args.model_name:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()

    # Validation
    with torch.no_grad():
        model.eval()
        dev_outputs = []
        dev_labels = []
        for _, (input_ids, attn_mask, _ , target_strings) in enumerate(tqdm(dev_loader)):
            
            # 
            outputs = model.generate(input_ids, 
                        tokenizer=train_set.tokenizer, 
                        )
            
            decoded_strings = outputs['decoded_strings']
            dev_outputs += decoded_strings
            dev_labels += target_strings
        
        
        assert len(dev_outputs) == len(dev_labels) , (len(dev_outputs), len(dev_labels))
        
        # model = model.cpu()
        dev_scores = compute_rouge(dev_outputs, dev_labels)
        # model = model.cuda()
        print(dev_scores)
        dev_f1 = dev_scores['avg_r']

        if dev_f1 > best_dev_f1:    
            print(f"Saving to {model_path}")
            best_dev_f1 = dev_f1
            torch.save(state, f'{model_path}')
        print(f"Epoch {epoch} dev F1: {dev_f1:.4f}. Best dev F1: {best_dev_f1:.4f}.")    


checkpoint = torch.load(model_path)
print(f"Loading from {model_path}")
model.load_state_dict(checkpoint['model'], strict=True)    
test_output_file = os.path.join(output_dir, 'test_pred.json')

with torch.no_grad():
    model.eval()
    test_outputs = []
    test_labels = []
    for _, (input_ids, attn_mask, _ , target_strings) in enumerate(test_loader):

        outputs = model.generate(input_ids, tokenizer=train_set.tokenizer)
        
        decoded_strings = outputs['decoded_strings']
        test_outputs += decoded_strings
        test_labels += target_strings
        
    
    
    assert len(test_outputs) == len(test_labels) , (len(test_outputs), len(test_labels))

    test_scores = compute_rouge(test_outputs, test_labels)
    print(test_scores)
    test_f1 = test_scores['avg_r']
    
    print(f"Epoch {epoch} test F1: {test_f1:.4f}. ")
    
    
    with open(test_output_file,'w') as f:
        json.dump({'output':test_outputs}, f)