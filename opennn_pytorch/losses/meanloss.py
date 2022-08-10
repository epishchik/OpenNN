from torch.nn import MSELoss, L1Loss
from torch import nn
import torch


def mse():
    '''
    Return mean-squared-error loss object.
    https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    '''
    return MSELoss()


def custom_mse():
    '''
    Return mean-squared-error loss object from CustomMSELoss class.
    https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    '''
    return CustomMSELoss()


def mae():
    '''
    Return mean-absolute-error loss object.
    https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
    '''
    return L1Loss()


def custom_mae():
    '''
    Return mean-absolute-error loss object from CustomMAELoss class.
    https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
    '''
    return CustomMAELoss()


class CustomMSELoss(nn.Module):
    '''
    Class used to calculate mean-squared-error loss.

    Attributes
    ----------
    reduction : str
        type of reduction ['mean', 'sum'].

    Methods
    -------
    forward(inputs, targets)
        calculate mean-squared-error loss between inputs and targets.
    '''
    def __init__(self, reduction='mean'):
        '''
        Parameters
        ----------
        reduction : str, optional
            type of reduction ['mean', 'sum'].
        '''
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        '''
        Calculate mean-squared-error loss between inputs and targets.

        Parameterts
        -----------
        inputs : torch.tensor
            model predictions.

        targets : torch.tensor
            ground-truth labels.
        '''
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        sub = (inputs - targets) ** 2
        if self.reduction == 'mean':
            return sub.mean()
        elif self.reduction == 'sum':
            return sub.sum()


class CustomMAELoss(nn.Module):
    '''
    Class used to calculate mean-absolute-error loss.

    Attributes
    ----------
    reduction : str
        type of reduction ['mean', 'sum'].

    Methods
    -------
    forward(inputs, targets)
        calculate mean-absolute-error loss between inputs and targets.
    '''
    def __init__(self, reduction='mean'):
        '''
        Parameters
        ----------
        reduction : str, optional
            type of reduction ['mean', 'sum'].
        '''
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        '''
        Calculate mean-absolute-error loss between inputs and targets.

        Parameterts
        -----------
        inputs : torch.tensor
            model predictions.

        targets : torch.tensor
            ground-truth labels.
        '''
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        sub = torch.abs(inputs - targets)
        if self.reduction == 'mean':
            return sub.mean()
        elif self.reduction == 'sum':
            return sub.sum()
