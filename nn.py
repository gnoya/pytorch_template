import torch.nn as nn

class NN(nn.Module):
    def __init__(self, model, optimizer, loss_function, lr_scheduler, config):
        super(NN, self).__init__()
        # Set up the model
        self.model = model

        # Set up the optimizer
        self.optimizer = optimizer(self.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        # Set up the loss function
        self.loss_function = loss_function()

        # Set up the learning rate scheduler
        self.lr_scheduler = lr_scheduler(self.optimizer, config['lr_scheduler']['milestones'], config['lr_scheduler']['gamma']) if lr_scheduler is not None else None
    
    def forward(self, x):
        return self.model(x)

    def backpropagate(self, y_pred, y):
        loss = self.loss_function(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()