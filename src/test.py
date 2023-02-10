from transformers import AdamW, get_linear_schedule_with_warmup
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
from evals import compute_rouge#, compute_bart_score
from args import create_args
from data import SummDataset
from constants import DIALOGSUM, SAMSUM


seed = 42
# fix random seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False


# configuration
args = create_args(is_training=False)

args.model_name = 'T5'



# TODO: read arg from file

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())


output_dir = '/'.join(args.checkpoint_path.split('/')[:-1])

# load config into args
with open(os.path.join(output_dir, 'config.json'), 'r') as f:
    configs = json.load(f)


for k, v in configs.items():
    if k != 'checkpoint_path':
        setattr(args, k, v)

# don't do gradient checkpointing during inference time
setattr(args, 'do_gradient_checkpointing', False)

# init model and tokenizer
if args.use_robust:
    model = RobustBART(args.model_name, args).cuda()
elif args.use_nli:
    model = NLIBART(args.model_name, args).cuda()
else:
    model = FCBART(args.model_name, args).cuda()


if args.dataset == SAMSUM:
    args.data_dir = '../data/samsum'
elif args.dataset == DIALOGSUM:
    args.data_dir = '../data/dialogsum/DialogSum_Data'
else:
    raise NotImplementedError
print(f'Reading data from {args.data_dir}')


test_path = os.path.join(args.data_dir, 'test.json')


test_set = SummDataset(test_path, args, is_test=True)
test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn)

state = dict(model=model.state_dict())

model_path = args.checkpoint_path

checkpoint = torch.load(model_path)
print(f"Loading from {model_path}")
model.load_state_dict(checkpoint['model'], strict=True)    
if args.output_name is None:
    test_output_file = os.path.join(output_dir, 'test_pred.json')
else:
    test_output_file = os.path.join(output_dir, args.output_name)

with torch.no_grad():
    model.eval()
    test_outputs = []
    test_labels = []
    for batch_idx, (input_ids, attn_mask, _ , target_strings) in enumerate(tqdm(test_loader)):

        outputs = model.generate(input_ids, tokenizer=test_set.tokenizer)
        
        decoded_strings = outputs['decoded_strings']
        test_outputs += decoded_strings
        test_labels += target_strings
        # if batch_idx > 5:
        #     break
    
    
    if not args.disable_eval:
        # move the model to cpu to save GPU memory
        model = model.cpu()
        assert len(test_outputs) == len(test_labels) , (len(test_outputs), len(test_labels))
        # if args.test_rouge:
        test_scores = compute_rouge(test_outputs, test_labels)
        print(test_scores)
        # test_scores = compute_bart_score(test_outputs, test_labels)
        # 
        # test_f1 = test_scores['avg_r']
        # print(test_scores)
        # print(f"Test F1: {test_f1:.4f}. ")
    
    print("saving to", test_output_file)
    with open(test_output_file,'w') as f:
        json.dump({'output':test_outputs}, f)