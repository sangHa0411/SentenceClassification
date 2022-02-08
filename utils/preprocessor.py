
class Preprocessor :
    def __init__(self, model_type, label_dict) :
        self.model_type = model_type
        self.label_dict = label_dict

    def __call__(self, dataset) :
        inputs = []
        labels = []

        size = len(dataset['index'])
        for i in range(size) :
            premise = dataset['premise'][i]
            hypothesis = dataset['hypothesis'][i]
            label = dataset['label'][i]
            
            input_sen = premise + ' [SEP] ' + hypothesis
            inputs.append(input_sen)
            labels.append(self.label_dict[label])

        dataset['inputs'] = inputs
        dataset['labels'] = labels
        return dataset
