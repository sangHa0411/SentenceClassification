
import datasets
from transformers import data


class Preprocessor :
    def __init__(self, label_dict, model_type) :
        self.label_dict = label_dict
        self.model_type = model_type

    def __call__(self, dataset) :
        if self.model_type == 'seq2seq' :
            inputs = []
            decoder_inputs = []
            labels = []

            size = len(dataset['index'])

            for i in range(size) :
                premise = dataset['premise'][i]
                hypothesis = dataset['hypothesis'][i]
                label = dataset['label'][i]

                inputs.append(premise)
                decoder_inputs.append(hypothesis)
                labels.append(label)
            
            dataset['inputs'] = inputs
            dataset['decoder_inputs'] = decoder_inputs
            dataset['labels'] = labels
            return dataset
        else :
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
