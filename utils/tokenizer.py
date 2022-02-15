
class Tokenizer :
    def __init__(self, tokenizer, max_input_length) :
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
    
    def encode4train(self, examples):
        inputs = examples['inputs']
        model_inputs = self.tokenizer(inputs, 
            max_length=self.max_input_length, 
            return_token_type_ids=False,
            truncation=True,
        )

        model_inputs['labels'] = examples['labels']
        return model_inputs

    def encode4test(self, examples):
        inputs = examples['inputs']
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)
        return model_inputs
