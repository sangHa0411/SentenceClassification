# DACON-NLI

## Goal
  1. Dacon 한국어 문자 관계 분류 경진대회 참가

## Dataset
  1. Dacon 
  2. 주소 : https://dacon.io/competitions/official/235875/overview/description
  3. 데이터 갯수 : 30000개
      
## Baseline 구축
  1. AutoModelForSequenceClassification, AutoConfig, AutoTokenizer을 이용한 대회에 맞는 모델 및 토크나이저 불러오기
  2. 다양하 모델을 설계하고 학습해서 성능으 비교하기
  3. 학습 과정 및 결과를 보기 위한 wandb 활용
  4. K-fold Validation 및 앙상블을 해보기

## Argument
|Argument|Description|Default|
|--------|-----------|-------|
|output_dir|model saving directory|exps|
|logging_dir|logging directory|logs|
|log_name|wandb log name|klue/roberta-large|
|PLM|model name(huggingface)|klue/roberta-large|
|model_type|custom model type|base|
|epochs|train epochs|5|
|lr|learning rate|2e-5|
|train_batch_size|train batch size|32|
|eval_batch_size|evaluation batch size|16|
|max_len|input max length|128|
|warmup_steps|warmup steps|500|
|weight_decay|weight decay|1e-3|
|evaluation_strategy|evaluation strategy|steps|
|gradient_accumulation_steps|gradient accumulation steps|1|
|save_steps|model save steps|300|
|logging_steps|logging steps|100|
|eval_steps|model evaluation steps|300|
|seed|random seed|42|
|dotenv_path|wandb & huggingface path|./path.env|


## Terminal
  ```
  # training 
  python train.py --PLM klue/roberta-large --lr 2e-5 --output_dir ./exps --epochs 5 --model_type base
  
  # K-fold training
  python train_kfold.py --PLM klue/roberta-large --k_fold 5 --lr 2e-5 --epochs 5 --model_type base --warmup_steps 300 --save_steps 500
  
  # inference 
  python inference.py --PLM /content/exp/checkpoint-3500 --tokenizer ./tokenizer --output_file ./results.csv
  ```

## Result
|Model|Accuracy|
|-----|----|
|ensemable|90.0|
