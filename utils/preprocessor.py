
class Preprocessor :
    def __init__(self, label_dict) :
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
