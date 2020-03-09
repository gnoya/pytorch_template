import torch.nn as nn
import torch

optimizer = torch.optim.Adam
loss_function = nn.MSELoss
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR

config = {
    'dataset': {
        'train_set': './dataset/shuffled_sonar.csv',
        'valid_set': None,
        'test_set': None,
        'train_set_len': 0.7,
        'valid_set_len': 0.3,
        'test_set_len': 0,
    },
    'epochs': 1000,
    'learning_rate': 0.005,
    'lr_scheduler': {
        'milestones': [100, 200, 300],
        'gamma': 0.1
    },
    'data_loader': {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 2
    },
    'weight_decay': 0
}