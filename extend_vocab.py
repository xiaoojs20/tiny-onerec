#!/usr/bin/env python3
import os
import json
from typing import List, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TokenExtender:
    """根据 index 文件扩展 tokenizer 词表"""
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens



def main(model_name_or_path, 
         sid_index_path,
         output_path,
         ):
    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--sid_index_path", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()
    main(args.model_name_or_path, args.sid_index_path, args.output_path)
    
"""
SFT_DATA_ROOT="./data/Amazon2018"
CATEGORY="Industrial_and_Scientific"
python extend_vocab.py \
    --model_name_or_path ../llms/Qwen/Qwen3-0.6B
    --sid_index_path "${SFT_DATA_ROOT}/${CATEGORY}/${CATEGORY}.index.json \
    --output_path "./rec_llm/Qwen3-0.6B-${CATEGORY}"
"""