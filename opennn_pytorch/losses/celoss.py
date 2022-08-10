from torch.nn import CrossEntropyLoss
from torch import nn
import torch


def celoss():
    '''
    Return cross-entropy loss object.
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    '''
    return CrossEntropyLoss()


def customceloss():
    '''
    Return custom cross-entropy loss object from CustomCrossEntropyLoss class.
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    '''
    return CustomCrossEntropyLoss()


class CustomCrossEntropyLoss(nn.Module):
    '''
    Class used to calculate cross-entropy loss.

    Attributes
    ----------
    weight : float, list[float]
        weight for loss value.

    reduction : str
        type of reduction ['mean', 'sum'].

    Methods
    -------
    forward(inputs, targets)
        calculate cross-entropy loss between inputs and targets.
    '''
    def __init__(self, weight=1.0, reduction='mean'):
        '''
        Parameters
        ----------
        weight : float, list[float], optional
            weight for loss value.

        reduction : str, optional
            type of reduction ['mean', 'sum'].
        '''
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        '''
        Calculate cross-entropy loss between inputs and targets.

        Parameterts
        -----------
        inputs : torch.tensor
            model predictions.

        targets : torch.tensor
            ground-truth labels.
        '''
        inputs = inputs.view(inputs.shape[0], -1)
        exps = torch.exp(torch.gather(inputs, 1, targets.view(-1, 1)))
        expinputs = torch.exp(inputs).sum()
        logs = torch.clamp(torch.log2(exps / expinputs), min=-100.0)
        loss = -self.weight * logs
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
