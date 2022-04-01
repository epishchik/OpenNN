import torch
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch import nn


def bce():
    return BCELoss()


def bcelogits():
    return BCEWithLogitsLoss()


def custombce():
    return CustomBCELoss()


def custombcelogits():
    return CustomBCEWithLogitsLoss()


class CustomBCELoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        logx = torch.clamp(torch.log2(inputs), min=-100.0)
        loginvx = torch.clamp(torch.log2(1.0 - inputs), min=-100.0)
        loss = -self.weight * (targets * logx + (1.0 - targets) * loginvx)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.sigm = nn.Sigmoid()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        sigm_inps = self.sigm(inputs)
        logx = torch.clamp(torch.log2(sigm_inps), min=-100.0)
        loginvx = torch.clamp(torch.log2(1.0 - sigm_inps), min=-100.0)
        loss = -self.weight * (targets * logx + (1.0 - targets) * loginvx)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
