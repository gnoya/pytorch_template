import torch.nn as nn
import torch
from sklearn import metrics

# Edit your parameters
optimizer = torch.optim.Adam
loss_function = nn.BCELoss
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
metric = metrics.f1_score

config = {
    'epochs': 1000,
    'learning_rate': 0.005,
    'lr_scheduler': {
        'milestones': [100, 200, 300],
        'gamma': 0.1
    },
    'dataset': {
        'train_set': './dataset/shuffled_sonar.csv',
        'valid_set': None,
        'test_set': None,
        'train_set_len': 0.8,
        'valid_set_len': 0.2,
        'test_set_len': 0,
    },
    'data_loader': {
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 2
    },
    'weight_decay': 0,
    'plot': {
        'loss': True,
        'metric': True
    },
    'save_path': 'checkpoint.pth.tar'
}