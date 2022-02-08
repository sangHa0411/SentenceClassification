
class Tokenizer :
    def __init__(self, tokenizer, max_input_length) :
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
    
    def __call__(self, examples):
        inputs = examples['inputs']
        model_inputs = self.tokenizer(inputs, 
            max_length=self.max_input_length, 
            return_token_type_ids=False,
            truncation=True,
        )

        model_inputs['labels'] = examples['labels']
        return model_inputs
