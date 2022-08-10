import torch
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch import nn


def bce():
    '''
    Return binary-cross-entropy loss object.
    https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    '''
    return BCELoss()


def bcelogits():
    '''
    Return binary-cross-entropy-with-logits loss object.
    https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    '''
    return BCEWithLogitsLoss()


def custombce():
    '''
    Return custom binary-cross-entropy loss object from CustomBCELoss class.
    https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    '''
    return CustomBCELoss()


def custombcelogits():
    '''
    Return custom binary-cross-entropy-with-logits loss object from CustomBCEWithLogitsLoss class.
    https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    '''
    return CustomBCEWithLogitsLoss()


class CustomBCELoss(nn.Module):
    '''
    Class used to calculate binary-cross-entropy loss.

    Attributes
    ----------
    weight : float, list[float]
        weight for loss value.

    reduction : str
        type of reduction ['mean', 'sum'].

    Methods
    -------
    forward(inputs, targets)
        calculate binary-cross-entropy loss between inputs and targets.
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
        Calculate binary-cross-entropy loss between inputs and targets.

        Parameterts
        -----------
        inputs : torch.tensor
            model predictions.

        targets : torch.tensor
            ground-truth labels.
        '''
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        logx = torch.clamp(torch.log2(inputs), min=-100.0, max=100.0)
        loginvx = torch.clamp(torch.log2(1.0 - inputs), min=-100.0, max=100.0)
        loss = -self.weight * (targets * logx + (1.0 - targets) * loginvx)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class CustomBCEWithLogitsLoss(nn.Module):
    '''
    Class used to calculate binary-cross-entropy-with-logits loss.

    Attributes
    ----------
    weight : float, list[float]
        weight for loss value.

    reduction : str
        type of reduction ['mean', 'sum'].

    sigm : Any
        sigmoid activation function.

    Methods
    -------
    forward(inputs, targets)
        calculate binary-cross-entropy-with-logits loss between inputs and targets.
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
        self.sigm = nn.Sigmoid()

    def forward(self, inputs, targets):
        '''
        Calculate binary-cross-entropy-with-logits loss between inputs and targets.

        Parameterts
        -----------
        inputs : torch.tensor
            model predictions.

        targets : torch.tensor
            ground-truth labels.
        '''
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        sigm_inps = self.sigm(inputs)
        logx = torch.clamp(torch.log2(sigm_inps), min=-100.0, max=100.0)
        loginvx = torch.clamp(torch.log2(1.0 - sigm_inps), min=-100.0, max=100.0)
        loss = -self.weight * (targets * logx + (1.0 - targets) * loginvx)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
