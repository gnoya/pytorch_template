import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from parameters import *

class Layer(nn.Module):
    def __init__(self, input, output, activation_id):
        super(Layer, self).__init__()

        self.input = input
        self.output = output
        self.activation_id = activation_id

        self.linear = nn.Linear(self.input, self.output)
        
        if activation_id == 'relu':
            self.activation = F.relu
        elif activation_id == 'tanh':
            self.activation = F.tanh
        elif activation_id == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation_id == 'leaky_relu':
            self.activation = torch.leaky_relu
        else:
            assert False, 'Incorrect activation function'

    def forward(self, x):
        return self.activation(self.linear(x))


class DQN(nn.Module):
    def __init__(self, layers_dicts, optimizer_id, batch_size, learning_rate):
        super(DQN, self).__init__()
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.batch_size = batch_size

        layer_list = []
        for l in layers_dicts:
            layer_list.append(Layer(l['input'], l['output'], l['act_id']))

        self.linear_net = nn.Sequential(*layer_list)
        
        if optimizer_id == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif optimizer_id == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer_id == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif optimizer_id == 'Momentum':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            assert False, 'Incorrect optimizer'

    def forward(self, x):
        return self.linear_net(x).view(x.size(0), -1)

    def backpropagate(self, y_pred, y):
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()