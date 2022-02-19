
class Tokenizer :
    def __init__(self, tokenizer, max_input_length, model_type) :
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.model_type = model_type
    
    def __call__(self, examples):
        inputs = examples['inputs']
        model_inputs = self.tokenizer(inputs, 
            max_length=self.max_input_length, 
            return_token_type_ids=False,
            truncation=True,
        )

        if self.model_type == 'seq2seq' :
            with self.tokenizer.as_target_tokenizer():
                target_inputs = self.tokenizer(examples["decoder_inputs"], max_length=self.max_input_length, return_token_type_ids=False, truncation=True)
            model_inputs['decoder_input_ids'] = target_inputs['input_ids']

        model_inputs['labels'] = examples['labels']
        return model_inputs