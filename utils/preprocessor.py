
class Preprocessor :
    def __init__(self, model_type, label_dict) :
        self.model_type = model_type
        self.label_dict = label_dict

    def bert_call(self, dataset) :
        labels = []
        size = len(dataset['index'])
        for i in range(size) :
            label = dataset['label'][i]
            labels.append(self.label_dict[label])
        dataset['labels'] = labels
        return dataset

    def roberta_call(self, dataset) :
        inputs = []
        labels = []

        size = len(dataset['index'])
        for i in range(size) :
            premise = dataset['premise'][i]
            hypothesis = dataset['hypothesis'][i]
            label = dataset['label'][i]

            input = premise + ' [SEP] ' + hypothesis
            inputs.append(input)
            labels.append(self.label_dict[label])

        dataset['inputs'] = inputs
        dataset['labels'] = labels
        return dataset

    def __call__(self, dataset) :
        if 'roberta' in self.model_type :
            return self.roberta_call(dataset)
        else :
            return self.bert_call(dataset)