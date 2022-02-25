import os
import copy
import torch
import argparse
import importlib
import pandas as pd
import numpy as np
import collections

from datasets import Dataset
from transformers import (AutoTokenizer, 
    AutoConfig, 
    AutoModelForSequenceClassification, 
    Trainer, 
    DataCollatorWithPadding
)

def inference(args):    
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
    def preprocess(dataset) :
        inputs = []
        size = len(dataset['index'])
        for i in range(size) :
            data = dataset['premise'][i] + ' [SEP] ' + dataset['hypothesis'][i]
            inputs.append(data)
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

    # -- Checkpoint Directory
    MODEL_NAME = args.PLM_DIR
    print('Model : %s' %MODEL_NAME)

    # -- Collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_len)

    size = len(test_df)

    ids_list = []
    prediction_list = []
    for i in range(args.k_fold) :
        print('%dth Inference' %(i+1))
        dir_path = MODEL_NAME + '_' + str(i+1)
        checkpoint_name = 'checkpoint-' + str(args.checkpoint)
        checkpoint_path = os.path.join(dir_path, checkpoint_name)

        # -- Config & Model
        print('\nLoad Model')
        config =  AutoConfig.from_pretrained(checkpoint_path)
        if args.model_type == 'base' :
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, config=config)
        else :
            model_type_str = 'models.' + args.model_type
            model_lib = importlib.import_module(model_type_str)
            model_class = getattr(model_lib, 'RobertaForSequenceClassification')
            model = model_class.from_pretrained(checkpoint_path, config=config)
        model = model.to(device)
    
        # -- Trainer
        trainer = Trainer(
            model=model, 
            data_collator=collator, 
        )

        # -- Inference
        results = trainer.predict(test_dataset=test_dset)
        prediction = results.predictions
        prediction_list.append(prediction)

        label_ids = np.argmax(prediction, axis=1)
        ids_list.append(label_ids)

    # -- Decoder
    print('\nDecoding Results')
    label2ids = {'contradiction' : 0, 'entailment' : 1, 'neutral' : 2}
    ids2label = {v:k for k,v in label2ids.items()}
    
    # -- Hard Voting
    voted_labels = []
    counter = collections.Counter()
    for i in range(size) :
        labels = [ids[i] for ids in ids_list]
        counter.update(labels)
        counter_dict = dict(counter)

        items = sorted(counter_dict.items(), key=lambda x : x[1], reverse=True)
        selected = ids2label[items[0][0]]
        voted_labels.append(selected)
        counter.clear()

    # -- Saving Hard voting Results
    voted_df = copy.deepcopy(test_df)
    voted_df = voted_df.drop(columns=['premise', 'hypothesis'])
    voted_df['label'] = voted_labels
    hard_file = os.path.join(args.output_dir, 'hardvoting.csv')
    voted_df.to_csv(hard_file, index=False)

    # -- Softmax Voting
    probs = np.sum(prediction_list, axis=0)
    labels = np.argmax(probs, axis=1)
    labels = [ids2label[v] for v in labels]

    # -- Saving Soft Voting Results
    print('\nSaving Results')
    voted_df = copy.deepcopy(test_df)
    voted_df['label'] = labels
    voted_df = voted_df.drop(columns=['premise', 'hypothesis'])
    softmax_file = os.path.join(args.output_dir, 'sotmaxvoting.csv')
    voted_df.to_csv(softmax_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- k fold
    parser.add_argument('--k_fold', type=int, default=5, help='the number of checkpoints (default: 5)')
    parser.add_argument('--checkpoint', type=int, default=3500, help='checkpoint number (default: 3500)')
    parser.add_argument('--PLM_DIR', type=str, default='./exp', help='base directory (default: ./exp)')

    # -- Result directory
    parser.add_argument('--output_dir', type=str, default='./', help='trained model output directory')

    # -- Dataset
    parser.add_argument('--dataset', type=str, default='./data/test_data.csv', help='test dataset directory')

    # -- Max input Length
    parser.add_argument('--max_len', type=int, default=128, help='input max length')

    # -- Model
    parser.add_argument('--model_type', type=str, default='base', help='custom model type (default: base)')
    parser.add_argument('--tokenizer', type=str, default='klue/roberta-large', help='model type (default: klue/roberta-large)')
    
    args = parser.parse_args()
    inference(args)

