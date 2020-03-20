import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# Edit this class in order to match your dataset
class CustomDataset(Dataset):
    def __init__(self):
        dataset = np.genfromtxt(config['dataset']['train_set'], delimiter=',', dtype=str)

        # Remove header and first column
        dataset = dataset[1:, 1:]

        x = dataset[:, :-1].astype(float)
        y = dataset[:, -1:]
        y[y == 'M'] = 1
        y[y == 'R'] = 0
        y = y.astype(float)

        self.length = y.shape[0]
        self.samples = torch.from_numpy(x).float()
        self.labels = torch.from_numpy(y).float()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.samples[index]
        y = self.labels[index]

        return x, y

    # This function works only if the whole dataset (every set) is in one file
    def get_loaders(self, config):
        samples = self.length
        training_set_len = int(config['dataset']['train_set_len'] * samples)

        if config['dataset']['test_set_len'] is 0:
            valid_set_len = samples - training_set_len
            training_set, validation_set = random_split(self, [training_set_len, valid_set_len])
            test_loader = None
        else:
            valid_set_len = int(config['dataset']['valid_set_len'] * samples)
            test_set_len = samples - training_set_len - valid_set_len
            training_set, validation_set, test_set = random_split(self, [training_set_len, valid_set_len, test_set_len])
            test_loader = DataLoader(dataset=test_set, **config['data_loader'])

        training_loader = DataLoader(dataset=training_set, **config['data_loader'])
        validation_loader = DataLoader(dataset=validation_set, **config['data_loader'])
    
        return training_loader, validation_loader, test_loader