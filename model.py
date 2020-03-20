import torch.nn as nn

# Edit your model
net = nn.Sequential(
    nn.Linear(60, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid()
)