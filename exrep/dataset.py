from functools import reduce
import operator

from torch.utils.data import Dataset

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
    def __len__(self):
        return len(self.hf_dataset)
    def __getitems__(self, indices):
        return [self[index] for index in indices]
    def __getitem__(self, index):
        # warning: this only works for scalar index
        return self.hf_dataset[index]
    
class StackDictDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.length = len(self.datasets[0])
        assert all(len(dataset) == self.length 
                for dataset in self.datasets), "Datasets need to have same length"
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return reduce(operator.ior, [
            dataset[index] for dataset in self.datasets
        ], {})