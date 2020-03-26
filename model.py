import torch.nn as nn

# Edit your model
net = nn.Sequential(
    nn.Linear(60, 256),
    nn.Dropout(0.25),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(256, 256),
    nn.Dropout(0.25),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(256, 1),
    nn.Sigmoid()
)