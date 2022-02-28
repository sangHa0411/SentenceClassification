import os
import wandb
import torch
import random
import optuna
import argparse
import numpy as np

from optuna import trial
from sklearn.metrics import f1_score
from datasets import load_dataset
from utils.optimizer import Optimizer
from utils.tokenizer import Tokenizer
from utils.preprocessor import Preprocessor

from dotenv import load_dotenv
from transformers import (AutoTokenizer, 
    AutoConfig, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    label_indices = list(range(3))
    f1 = f1_score(labels, preds, average="micro", labels=label_indices) * 100.0
    return {'micro f1 score': f1}

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-5),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5, 1),
        "per_device_train_batch_size": trial.suggest_int("train_batch_size", 8, 32, 4),
        "warmup_steps": trial.suggest_int("warmup_steps", 100, 500, 100),
        "weight_decay": trial.suggest_loguniform("weight_decay", low=1e-4, high=1e-3),
    }

def search(args):
    # -- Checkpoint 
    MODEL_NAME = args.PLM
    print('Model : %s' %MODEL_NAME)

    # -- Loading Dataset
    print('\nLoad Dataset')
    dset = load_dataset('sh110495/klue-nli')
    train_dset, validation_dset = dset['train'], dset['validation']
    train_dset = train_dset.shuffle(seed=args.seed)
    validation_dset = validation_dset.shuffle(seed=args.seed)
    
    # -- Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -- Config & Model
    config =  AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 3

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', config=config).to(device)
        
    # -- Tokenizing Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -- Optmizer 
    print('\nOptimize Tokenizer')
    if os.path.exists('./data/unk_chars.csv') :
        optimizer = Optimizer(tokenizer, './tokenizer', './data/unk_chars.csv')
        tokenizer = optimizer.load()

    # -- Preprocessing Dataset
    print('\nPreprocess Dataset')
    label_dict = {'contradiction' : 0, 'entailment' : 1, 'neutral' : 2}
    preprocessor = Preprocessor(label_dict=label_dict)
    train_dset = train_dset.map(preprocessor, batched=True)
    validation_dset = validation_dset.map(preprocessor, batched=True)
   
    print('\nEncoding Dataset')
    convertor = Tokenizer(tokenizer=tokenizer, max_input_length=args.max_len)

    print('Trainin Dataset')
    train_dset = train_dset.map(convertor, batched=True, remove_columns=train_dset.column_names)
    print(train_dset)
    print('Validation Dataset')
    validation_dset = validation_dset.map(convertor, batched=True, remove_columns=validation_dset.column_names)
    print(validation_dset)

    # -- Training Argument
    training_args = TrainingArguments(output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        save_strategy='no',
        evaluation_strategy='steps',
        eval_steps=args.eval_steps, 
        logging_steps=args.logging_steps,
        report_to='wandb'
    )

    # -- Collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_len)

    # -- Trainer
    trainer = Trainer(
        model_init=model_init,               # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dset,            # training dataset
        eval_dataset=validation_dset,        # evaluation dataset
        data_collator=collator,              # collator
        compute_metrics=compute_metrics      # define metrics function
    )

    # -- Hyperparameter Search
    trainer.hyperparameter_search(
        direction="maximize",          # NOTE: or direction="minimize"
        hp_space=optuna_hp_space,      # NOTE: if you wanna use optuna, change it to optuna_hp_space
        n_trials=args.trials,
    )


def main(args):
    load_dotenv(dotenv_path=args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    wandb_name = args.log_name
    wandb.init(entity="sangha0411", project="dacon - NLU", name=wandb_name, group='hyperparameter search')
    wandb.config.update(args)
    search(args)
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

    # -- Directory
    parser.add_argument('--output_dir', default='./exp', type=str, help='trained model output directory')
    parser.add_argument('--logging_dir', default='./logs', type=str, help='logging directory')

    # -- Trials 
    parser.add_argument('--trials', default=15, type=int, help='logging directory')
    
    # -- PLM
    parser.add_argument('--PLM', default='klue/roberta-large', type=str, help='model type (default: klue/roberta-large)')

    # -- Length
    parser.add_argument('--max_len', type=int, default=128, help='max length of tensor (default: 128)')

    # -- Logging & Evaluation steps
    parser.add_argument('--log_name', default='klue/roberta-large', type=str, help='wandb logging name')
    parser.add_argument('--logging_steps', default=100, type=int, help='training log steps')
    parser.add_argument('--eval_steps', default=300, type=int, help='evaluation steps')

    # -- Seed
    parser.add_argument('--seed', default=42, type=int, help='random seed (default: 42)')

    # -- Wandb
    parser.add_argument('--dotenv_path', default='./path.env', type=str, help='input your dotenv path')

    args = parser.parse_args()

    seed_everything(args.seed)   
    main(args)

