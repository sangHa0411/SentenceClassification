import os
import torch
import argparse
import importlib
import pandas as pd
import numpy as np
from functools import partial
from datasets import Dataset
from utils.collator import DataCollatorForSeq2Seq
from transformers import (AutoTokenizer, 
    AutoConfig, 
    AutoModelForSequenceClassification, 
    Trainer, 
    DataCollatorWithPadding
)

def preprocess(dataset, model_type) :
    if model_type == 'seq2seq' :
        inputs = []
        decoder_inputs = []
        size = len(dataset['index'])
        for i in range(size) :
            inputs.append(dataset['premise'][i])
            decoder_inputs.append(dataset['hypothesis'][i])
        dataset['inputs'] = inputs
        dataset['decoder_inputs'] = decoder_inputs
        return dataset
    else :
        inputs = []
        size = len(dataset['index'])
        for i in range(size) :
            data = dataset['premise'][i] + ' [SEP] ' + dataset['hypothesis'][i]
            inputs.append(data)
        dataset['inputs'] = inputs
        return dataset

def convert(examples, tokenizer, max_len, model_type) :
    inputs = examples['inputs']
    model_inputs=tokenizer(inputs, max_length=max_len, truncation=True)

    if model_type == 'seq2seq' :
        with tokenizer.as_target_tokenizer():
            target_inputs = tokenizer(examples["decoder_inputs"], max_length=max_len, return_token_type_ids=False, truncation=True)
        model_inputs['decoder_input_ids'] = target_inputs['input_ids']
    return model_inputs

def inference(args):
    # -- Checkpoint 
    MODEL_NAME = args.PLM
    print('Model : %s' %MODEL_NAME)
    
    # -- Loading Dataset
    print('\nLoad Dataset')
    test_df = pd.read_csv(args.dataset)
    test_dset = Dataset.from_pandas(test_df)    
    print(test_dset)
    
    # -- Tokenizer
    print('\nLoad Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # -- Preprocessing Dataset
    print('\nPreprocessing Dataset')
    preprocess_fn = partial(preprocess, model_type=args.model_type)
    test_dset = test_dset.map(preprocess_fn, batched=True)

    # -- Converting Dataset
    print('\nConverting Dataset')
    convert_fn = partial(convert, tokenizer=tokenizer, max_len=args.max_len, model_type=args.model_type)
    test_dset = test_dset.map(convert_fn, 
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
    if args.model_type == 'base' :
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    else :
        model_type_str = 'models.' + args.model_type
        model_lib = importlib.import_module(model_type_str)
        model_class = getattr(model_lib, 'RobertaForSequenceClassification')

        if args.model_type == 'seq2seq' :
            model_size = 'klue/roberta-large' if args.model_size == 'large' else 'klue/roberta-base'
            model = model_class(model_size, config=config)
            model_parameters = os.path.join(MODEL_NAME, 'pytorch_model.bin')
            model.load_state_dict(torch.load(model_parameters)) # load model parameter
        else :
            model = model_class.from_pretrained(MODEL_NAME, config=config) 

    # -- Setting Device 
    model = model.to(device)
    
    # -- Collator
    if args.model_type == 'seq2seq' :
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=args.max_len)
    else :
        collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_len)

    # -- Trainer
    trainer = Trainer(
        model=model, 
        data_collator=collator, 
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
    test_df.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- Result directory
    parser.add_argument('--output_file', type=str, default='./result_data.csv', help='trained model output directory')

    # -- Dataset
    parser.add_argument('--dataset', type=str, default='./data/test_data.csv', help='test dataset directory')

    # -- Max input Length
    parser.add_argument('--max_len', type=int, default=128, help='input max length')

    # -- Model
    parser.add_argument('--model_size', type=str, default='large', help='model size (default : large)')
    parser.add_argument('--model_type', type=str, default='base', help='custom model type (default: base)')
    parser.add_argument('--PLM', type=str, default='klue/roberta-large', help='model type (default: klue/roberta-large)')
    parser.add_argument('--tokenizer', type=str, default='klue/roberta-large', help='model type (default: klue/roberta-large)')
    
    args = parser.parse_args()
    inference(args)

