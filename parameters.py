import torch.nn as nn
import torch
from sklearn import metrics

# Edit your parameters
optimizer = torch.optim.Adam
loss_function = nn.BCELoss
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
metric = metrics.f1_score

config = {
    'epochs': 250,
    'learning_rate': 0.002,
    'lr_scheduler': {
        'milestones': [150, 200],
        'gamma': 0.333
    },
    'dataset': {
        # If the dataset is in just one file. Default values: None
        'whole_set': './dataset/shuffled_sonar.csv',
        'train_set_len': 0.7,
        'valid_set_len': 0.3,
        'test_set_len': 0,
        # If the dataset is in multiple files. Default values: None
        'train_set': None,
        'valid_set': None,
        'test_set': None
    },
    'data_loader': {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 2
    },
    'weight_decay': 0.001,
    'plot': {
        'loss': True,
        'metric': True
    },
    'save_path': './checkpoint.pth.tar'
}