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
import argparse


from evals import compute_rouge, compute_bart_score, compute_factcc_score, compute_qafacteval_score
from args import create_args
from data import SummDataset
from constants import DIALOGSUM, SAMSUM
from utils import load_data

seed = 42
# fix random seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=[DIALOGSUM, SAMSUM])    
parser.add_argument('--output_file', required=True)    
# parser.add_argument('--do_qafacteval', action='store_true')
args = parser.parse_args()


if args.dataset == SAMSUM:
    args.data_dir = '../data/samsum'
elif args.dataset == DIALOGSUM:
    args.data_dir = '../data/dialogsum/DialogSum_Data'
else:
    raise NotImplementedError
print(f'Reading data from {args.data_dir}')


test_path = os.path.join(args.data_dir, 'test.json')
test_gold = load_data(test_path, is_test=True, args=args)
golds = test_labels = [sample['summary'] if args.dataset == SAMSUM else sample['summaries'] for sample in test_gold]

test_output_file = args.output_file


test_outputs = json.load(open(test_output_file,'r'))
preds = test_outputs = test_outputs['output']


# if args.test_rouge:
test_scores = compute_rouge(test_outputs, test_labels)
print(test_scores)

# if args.do_qafacteval:
#     test_scores = compute_qafacteval_score(test_outputs, test_labels)
#     print(f"QAFactEval score {test_scores:.2f}")
# else:

bart_score = compute_bart_score(preds, golds, reverse=False)
print("BART score", bart_score)

reverse_bart_score = compute_bart_score(preds, golds, reverse=True)
print("Reverse BART score", reverse_bart_score)

print("F1 BART score", (bart_score['avg_r'] + reverse_bart_score['avg_r'])/2)

# test_f1 = test_scores['avg_r']
# print("BART score", test_scores)


factcc_score = compute_factcc_score(preds, golds, reverse=False)
print(f"FactCC score {factcc_score*100:.2f}")

reverse_factcc_score = compute_factcc_score(preds, golds, reverse=True)
print(f"Reverse FactCC score {reverse_factcc_score*100:.2f}")

print(f"F1 FactCC score {(factcc_score + reverse_factcc_score)/2 * 100:.2f}")