import os
import wandb
import torch
import random
import argparse
import pandas as pd
import numpy as np

from datasets import Dataset
from model import RobertaForSequenceClassification
from transformers import (AutoTokenizer, 
    AutoConfig, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)

def inference(args):
    # -- Checkpoint 
    MODEL_NAME = args.PLM
    print('Model : %s' %MODEL_NAME)
    
    # -- Loading Dataset
    print('\nLoad Dataset')
    test_df = pd.read_csv(args.dataset)
    test_dset = Dataset.from_pandas(test_df)    
    
    # -- Tokenizer
    print('\nLoad Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # -- Preprocessing Dataset
    print('\nPreprocessing Dataset')
    def preprocess(dataset) :
        inputs = []
        size = len(dataset['index'])
        for i in range(size) :
            data = dataset['premise'][i] + ' [SEP] ' + dataset['hypothesis'][i]
            inputs.append(input)
        dataset['inputs'] = inputs
        return dataset
    test_dset = test_dset.map(preprocess, batched=True)

    # -- Converting Dataset
    print('\nConverting Dataset')
    def convert(examples, tokenizer, max_len) :
        inputs = examples['inputs']
        model_inputs=tokenizer(inputs, max_length=max_len, truncation=True)
        return model_inputs
    test_dset = test_dset.map(lambda x : convert(x, tokenizer=tokenizer, max_len=args.max_len), 
        batched=True, 
        remove_columns=test_dset.column_names
    )
    print(test_dset)

    # -- Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # -- Config & Model
    print('\nLoad Model')
    config =  AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config = config).to(device)

    # -- Collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_len)

    # -- Trainer
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        data_collator=collator,              # collator
    )

    # -- Inference
    results = trainer.predict(test_dataset=test_dset)

    # -- Decoding Results
    print('\nDecoding Results')
    labels = np.argmax(results.predictions, axis=1)
    label2ids = {'contradiction' : 0, 'entailment' : 1, 'neutral' : 2}
    ids2label = {v:k for k,v in label2ids.items()}
    labels = [ids2label[v] for v in labels]

    # -- Saving Results
    print('\nSaving Results')
    test_df['label'] = labels
    test_df = test_df.drop(columns=['premise', 'hypothesis'])
    test_df.to_csv(args.output_dir, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- Result directory
    parser.add_argument('--output_dir', type=str, default='./results', help='trained model output directory')

    # -- Dataset
    parser.add_argument('--dataset', type=str, default='./data/test_data.csv', help='test dataset directory')

    # -- Max input Length
    parser.add_argument('--max_len', type=int, default=128, help='input max length')

    # -- Model
    parser.add_argument('--PLM', type=str, default='klue/roberta-large', help='model type (default: klue/roberta-large)')
    parser.add_argument('--tokenizer', type=str, default='klue/roberta-large', help='model type (default: klue/roberta-large)')
    
    inference(args)

