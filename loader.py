import pandas as pd
from datasets import Dataset

class Loader : 
    def __init__(self, train_path, validation_path) :
        self.train_dataset = pd.read_csv(train_path)
        self.validation_dataset = pd.read_csv(validation_path)

    def get_data(self) :
        train_dset = Dataset.from_pandas(self.train_dataset)
        validation_dset = Dataset.from_pandas(self.validation_dataset)
        return train_dset, validation_dset
