import torch
from torch import nn


class CustomBCELoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
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
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.sigm = nn.Sigmoid()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        sigm_inps = self.sigm(inputs)

        logx = torch.clamp(torch.log2(sigm_inps), min=-100.0, max=100.0)
        loginvx = torch.clamp(torch.log2(1.0 - sigm_inps),
                              min=-100.0, max=100.0)

        loss = -self.weight * (targets * logx + (1.0 - targets) * loginvx)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(inputs.shape[0], -1)

        exps = torch.exp(torch.gather(inputs, 1, targets.view(-1, 1)))
        expinputs = torch.exp(inputs).sum()

        logs = torch.clamp(torch.log2(exps / expinputs), min=-100.0)
        loss = -self.weight * logs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


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


class CustomL1Loss(nn.Module):
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
