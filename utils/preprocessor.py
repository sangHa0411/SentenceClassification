
class Preprocessor :
    def __init__(self, label_dict) :
        self.label_dict = label_dict

    def preprocess4train(self, dataset) :
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

    def preprocess4test(self, dataset) :
        inputs = []
        size = len(dataset['index'])
        for i in range(size) :
            data = dataset['premise'][i] + ' [SEP] ' + dataset['hypothesis'][i]
            inputs.append(data)
        dataset['inputs'] = inputs
        return dataset