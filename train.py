import os
import wandb
import torch
import random
import argparse
import numpy as np

from loader import Loader
from optimizer import Optimizer
from tokenizer import Tokenizer
from preprocessor import Preprocessor

from dotenv import load_dotenv
from metrics import compute_metrics
from transformers import (AutoTokenizer, 
    AutoConfig, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)

def train(args):
    # -- Checkpoint 
    MODEL_NAME = args.PLM
    print('Model : %s' %MODEL_NAME)
    
    # -- Loading Dataset
    print('\nLoad Dataset')
    loader = Loader('./data/train_data.csv', './data/dev_data.csv')
    train_dset, validation_dset = loader.get_data()

    # -- Tokenizing Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -- Optmizer 
    print('\nOptimize Tokenizer')
    optimizer = Optimizer(tokenizer, './tokenizer', './data/unk_chars.csv')
    tokenizer = optimizer.load()

    # -- Preprocessing Dataset
    print('\nPreprocess Dataset')
    label_dict = {'contradiction' : 0, 'entailment' : 1, 'neutral' : 2}
    preprocessor = Preprocessor(model_type=args.PLM, label_dict=label_dict)

    train_dset = train_dset.map(preprocessor, batched=True)
    validation_dset = validation_dset.map(preprocessor, batched=True)

    print('\nEncoding Dataset')
    convertor = Tokenizer(model_type=args.PLM, tokenizer=tokenizer, max_input_length=args.max_len)
    print('Trainin Dataset')
    train_dset = train_dset.map(convertor, batched=True, remove_columns=train_dset.column_names)
    print(train_dset)

    print('Validation Dataset')
    validation_dset = validation_dset.map(convertor, batched=True, remove_columns=validation_dset.column_names)
    print(validation_dset)

    # -- Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -- Config & Model
    config =  AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 3

    print('\nLoad Model')
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config = config).to(device)

    # -- Training Argument
    training_args = TrainingArguments(
        output_dir=args.output_dir,                         # output directory
        save_total_limit=5,                                 # number of total save model.
        save_steps=args.save_steps,                         # model saving step.
        num_train_epochs=args.epochs,                       # total number of training epochs
        learning_rate=args.lr,                              # learning_rate
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,    # batch size for evaluation
        warmup_steps=args.warmup_steps,                     # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,                     # strength of weight decay
        logging_dir=args.logging_dir,                       # directory for storing logs
        logging_steps=args.logging_steps,                   # log saving step.
        evaluation_strategy=args.evaluation_strategy,       # evaluation strategy to adopt during training
        eval_steps=args.eval_steps,                         # evaluation step.
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end = True,
        report_to='wandb'
    )

    # -- Collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_len)

    # -- Trainer
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dset,            # training dataset
        eval_dataset=validation_dset,        # evaluation dataset
        data_collator=collator,              # collator
        compute_metrics=compute_metrics      # define metrics function
    )

    # -- Training
    print('Training Strats')
    trainer.train()

    print('Evaluation Starts')
    trainer.evaluate()

def main(args):
    load_dotenv(dotenv_path=args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    wandb_name = f"PLM:{args.PLM}_epochs:{args.epochs}_lr:{args.lr}"
    wandb.init(
        entity="sangha0411",
        project="daycon - NLU", 
        name=wandb_name,
        group='TRAIN')

    wandb.config.update(args)
    train(args)
    wandb.finish()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- directory
    parser.add_argument('--output_dir', default='./exp', help='trained model output directory')
    parser.add_argument('--logging_dir', default='./logs', help='logging directory')

    # -- plm
    parser.add_argument('--PLM', type=str, default='klue/bert-base', help='model type (default: klue/bert-base)')

    # -- training arguments
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate (default: 3e-5)')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train (default: 3)')
    parser.add_argument('--train_batch_size', type=int, default=32, help='train batch size (default: 32)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='strength of weight decay (default: 1e-4)')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='number of warmup steps for learning rate scheduler (default: 1000)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps (default: 1)')


    # -- validation arguments
    parser.add_argument('--validation_ratio', type=float, default=0.2, help='validation data ratio')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='eval batch size (default: 16)')
    parser.add_argument('--max_len', type=int, default=128, help='max length of tensor (default: 128)')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='evaluation strategy to adopt during training, steps or epoch (default: steps)')
    
    # -- save & log
    parser.add_argument('--save_steps', type=int, default=500, help='model save steps')
    parser.add_argument('--logging_steps', type=int, default=100, help='training log steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='evaluation steps')

    # -- Seed
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    # -- Wandb
    parser.add_argument('--dotenv_path', default='./path.env', help='input your dotenv path')

    args = parser.parse_args()

    seed_everything(args.seed)   
    main(args)

