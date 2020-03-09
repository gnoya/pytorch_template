import torch.nn as nn
import torch
from parameters import *
from model import net
from nn import NN
from dataset import CustomDataset

def run():
    # Create dataset
    custom_dataset = CustomDataset(config['dataset']['train_set'])
    training_loader, validation_loader, test_loader = custom_dataset.get_loaders(config)

    model = NN(net, optimizer, loss_function, lr_scheduler, config)

    for epoch in range(config['epochs']):
        for i, data in enumerate(training_loader, 0):
            x, y = data
            y_hat = model(x)
            loss = model.backpropagate(y_hat, y)
            print(loss)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    run()