import os
import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class KazakhSFTDataset(Dataset):
    """
    Supervised Fine-Tuning (SFT) Dataset for YABGU model.
    Applies prompt masking so the model only learns to predict the assistant's response.
    """
    def __init__(self, data_path_or_list, tokenizer_path, max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Load samples
        if isinstance(data_path_or_list, list):
            self.samples = data_path_or_list
        elif isinstance(data_path_or_list, str):
            self.samples = []
            if data_path_or_list.endswith(".jsonl"):
                with open(data_path_or_list, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.samples.append(json.loads(line))
            elif data_path_or_list.endswith(".json"):
                with open(data_path_or_list, "r", encoding="utf-8") as f:
                    self.samples = json.load(f)
        
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>") or 3
        self.eos_token_id = self.tokenizer.token_to_id("<|eos|>") or 2
        self.bos_token_id = self.tokenizer.token_to_id("<|bos|>") or 1
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        instruction = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        response = item.get("output", item.get("response", "")).strip()
        
        if inp:
            prompt_str = f"<|bos|>### Сұрақ:\n{instruction}\n\nҚосымша мәлімет:\n{inp}\n\n### Жауап:\n"
        else:
            prompt_str = f"<|bos|>### Сұрақ:\n{instruction}\n\n### Жауап:\n"
            
        full_str = f"{prompt_str}{response}<|eos|>"
        
        prompt_ids = self.tokenizer.encode(prompt_str).ids
        full_ids = self.tokenizer.encode(full_str).ids
        
        prompt_len = len(prompt_ids)
        
        # Truncate if exceeds max_seq_len
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]
            if full_ids[-1] != self.eos_token_id:
                full_ids[-1] = self.eos_token_id
                
        seq_len = len(full_ids)
        
        # Prepare inputs and targets (labels)
        # Shifted targets are handled in loss computation, or standard PyTorch causal LM format:
        # input_ids: full_ids[:-1]
        # labels: full_ids[1:]
        input_ids = full_ids[:-1]
        labels = list(full_ids[1:])
        
        # Mask out prompt tokens so loss is ONLY computed on the response
        # Prompt length in input_ids is prompt_len - 1 (since first token predicts second)
        mask_len = min(prompt_len - 1, len(labels))
        for i in range(mask_len):
            labels[i] = -100
            
        # Pad up to max_seq_len - 1
        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            
        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }
