import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, path):
        # Edit this function
        dataset = np.genfromtxt(path, delimiter=',', dtype=str)
        dataset = dataset[1:, 1:]

        x = dataset[:, :-1].astype(float)
        y = dataset[:, -1:]
        y[y == 'M'] = 1
        y[y == 'R'] = 0
        y = y.astype(float)

        # End of editing
        self.length = y.shape[0]
        self.samples = torch.from_numpy(x).float()
        self.labels = torch.from_numpy(y).float()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.samples[index]
        y = self.labels[index]

        return x, y

class TemplateDataset():
    def __init__(self, config):
        self.create_datasets(config)
        self.create_loaders(config)

    def create_datasets(self, config):
        if config['dataset']['whole_set'] is not None:
            assert config['dataset']['train_set_len'] + config['dataset']['valid_set_len'] + config['dataset']['test_set_len'] == 1, "Sum of dataset lengths must be 1"
            self.whole_dataset(config)
        else:
            self.splitted_dataset(config)

    def whole_dataset(self, config):
        # TODO: optimize this function
        # TODO: what if only training set?
        # TODO: what if no training set?
        initial_dataset = CustomDataset(config['dataset']['whole_set'])
        samples = initial_dataset.length
        training_set_len = int(config['dataset']['train_set_len'] * samples)

        if config['dataset']['test_set_len'] == 0:
            valid_set_len = samples - training_set_len
            self.training_set, self.validation_set = random_split(initial_dataset, [training_set_len, valid_set_len])
            self.testing_set = None
        else:
            valid_set_len = int(config['dataset']['valid_set_len'] * samples)
            test_set_len = samples - training_set_len - valid_set_len
            self.training_set, self.validation_set, self.testing_set = random_split(initial_dataset, [training_set_len, valid_set_len, test_set_len])

    def splitted_dataset(self, config):
        self.training_set = CustomDataset(config['dataset']['train_set']) if config['dataset']['train_set'] is not None else None
        self.validation_set = CustomDataset(config['dataset']['valid_set']) if config['dataset']['valid_set'] is not None else None
        self.testing_set = CustomDataset(config['dataset']['test_set']) if config['dataset']['test_set'] is not None else None

    def create_loaders(self, config):
        self.training_loader = DataLoader(dataset=self.training_set, **config['data_loader']) if self.training_set is not None else None
        self.validation_loader = DataLoader(dataset=self.validation_set, **config['data_loader']) if self.validation_set is not None else None
        self.testing_loader = DataLoader(dataset=self.testing_set, **config['data_loader']) if self.testing_set is not None else None

    def get_loaders(self):
        return self.training_loader, self.validation_loader, self.testing_loader