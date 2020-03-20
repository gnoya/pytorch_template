import torch.nn as nn
import torch
from model import net
from nn import NN
from dataset import CustomDataset
from data_handler import DataHandler
from parameters import *

def run():
    # Create dataset
    custom_dataset = CustomDataset()
    training_loader, validation_loader, test_loader = custom_dataset.get_loaders(config)

    # Create the neural network
    model = NN(net, optimizer, loss_function, lr_scheduler, config)

    # Create the data handler
    data_handler = DataHandler(training_loader is not None, validation_loader is not None, test_loader is not None)

    for epoch in range(config['epochs']):
        # Training
        model.train()
        for i, data in enumerate(training_loader, 0):
            x, y = data
            y_hat = model(x)
            loss = model.backpropagate(y_hat, y)
            metric = model.evaluate(y_hat, y)
            data_handler.train_loss.append(loss)
            data_handler.train_metric.append(metric)

        with torch.no_grad():
            # Validating
            model.eval()
            for i, data in enumerate(validation_loader, 0):
                x, y = data
                y_hat = model(x)
                _, loss = model.calculate_loss(y_hat, y)
                metric = model.evaluate(y_hat, y)
                data_handler.valid_loss.append(loss)
                data_handler.valid_metric.append(metric)

            # Testing
            if test_loader is not None:
                for i, data in enumerate(test_loader, 0):
                    x, y = data
                    y_hat = model(x)
                    _, loss = model.calculate_loss(y_hat, y)
                    metric = model.evaluate(y_hat, y)
                    data_handler.test_loss.append(loss)
                    data_handler.test_metric.append(metric)

        model.lr_scheduler_step()
        data_handler.epoch_end(epoch, model.get_lr())
        if epoch == 5:
            data_handler.plot(loss=config['plot']['loss'], metric=config['plot']['metric'])
    data_handler.plot(loss=config['plot']['loss'], metric=config['plot']['metric'])

if __name__ == '__main__':
    # torch.multiprocessing.freeze_support()
    try:
        run()
    except KeyboardInterrupt:
        # Save model here
        pass