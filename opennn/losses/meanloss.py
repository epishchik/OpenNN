from torch.nn import MSELoss, L1Loss
from torch import nn
import torch


def mse():
    return MSELoss()


def custom_mse():
    return CustomMSELoss()


def mae():
    return L1Loss()


def custom_mae():
    return CustomMAELoss()


class CustomMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        sub = (inputs - targets) ** 2
        if self.reduction == 'mean':
            return sub.mean()
        elif self.reduction == 'sum':
            return sub.sum()


class CustomMAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        sub = torch.abs(inputs - targets)
        if self.reduction == 'mean':
            return sub.mean()
        elif self.reduction == 'sum':
            return sub.sum()
