import os
import json
import pandas as pd
from transformers import AutoTokenizer

class Optimizer :
    def __init__(self, tokenizer, dir_path, char_path) :
        self.tokenizer = tokenizer
        self.dir_path = dir_path

        char_df = pd.read_csv(char_path)
        self.chars = list(char_df['UNK'])

    def get_data(self) :
        self.tokenizer.save_pretrained(self.dir_path)
        path = os.path.join(self.dir_path, 'tokenizer.json')

        with open(path, 'r') as f :
            data = json.load(f)
        return data

    def add_chars(self, data, char_list) :
        token2ids = data['model']['vocab']
        ids2token = {v:k for k,v in token2ids.items()}

        char_size = len(char_list)
        unused_ids = token2ids['[unused0]']

        for i in range(unused_ids, unused_ids + char_size) :
            ids = i - unused_ids
            ids2token[i] = char_list[ids]        
        
        chars = list(ids2token.values())
        f = open(os.path.join(self.dir_path, 'vocab.txt'), 'w')
        for i in range(len(chars)):
            f.write(chars[i]+'\n')
        f.close()

        data['model']['vocab'] = {v:k for k,v in ids2token.items()}
        path = os.path.join(self.dir_path, 'tokenizer.json')
        with open(path, 'w') as f :
            json.dump(data, f)

    def load(self) :
        data = self.get_data()
        self.add_chars(data, self.chars)
        
        tokenizer = AutoTokenizer.from_pretrained(self.dir_path)
        return tokenizer
