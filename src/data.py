import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import pandas as pd
from utils import load_data
from constants import DIALOGSUM, SAMSUM

class SummDataset(Dataset):
    def __init__(self, json_path, args, is_test=False):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        self.data = []
        # do not 
        
        data = load_data(json_path, is_test, args)
        
        for sample in data:
            src = sample['dialogue']
            if args.dataset == DIALOGSUM and is_test:
                # set it to 1 so that it works
                
                tgt = sample['summary1']
                tgt_string = sample['summaries']
            else:
                tgt = tgt_string = sample['summary']
            
            inputs = self.tokenizer(src, max_length=args.max_sequence_length, padding="max_length", truncation=True)
            
            
            decoder_input_output_ids = self.tokenizer.encode(tgt, max_length=128, padding="max_length", truncation=True)

            if 'bart' in args.model_name:
                decoder_input_output_ids = [self.tokenizer.eos_token_id] + decoder_input_output_ids

            self.data.append({
                'input_ids':inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'decoder_input_output_ids': decoder_input_output_ids,
                'target_strings': tgt_string,
            })
            
    

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]['input_ids'], self.data[idx]['attention_mask'], self.data[idx]['decoder_input_output_ids'], self.data[idx]['target_strings']
    
    def collate_fn(self, batch):
        # print(batch)
        input_ids = torch.cuda.LongTensor([inst[0] for inst in batch])
        attention_masks = torch.cuda.LongTensor([inst[1]for inst in batch])
        decoder_input_output_ids = torch.cuda.LongTensor([inst[2]for inst in batch])
        target_strings = [inst[3] for inst in batch]
        
        return input_ids, attention_masks, decoder_input_output_ids, target_strings