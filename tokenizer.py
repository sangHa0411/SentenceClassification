
class Tokenizer :
    def __init__(self, model_type, tokenizer, max_input_length) :
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length


    def bert_call(self, examples) :
        premise = examples['premise']
        hypothesis = examples['hypothesis']

        model_inputs = self.tokenizer(premise, 
            hypothesis,
            max_length=self.max_input_length, 
            truncation=True,
        )

        model_inputs["labels"] = examples["labels"]
        return model_inputs
    
    def roberta_call(self, examples) :
        inputs = examples['inputs']
        model_inputs = self.tokenizer(inputs, 
            max_length=self.max_input_length, 
            truncation=True,
        )

        model_inputs["labels"] = examples["labels"]
        return model_inputs

    def __call__(self, examples):
        if 'roberta' in self.model_type :
            return self.roberta_call(examples)
        else :
            return self.bert_call(examples)
