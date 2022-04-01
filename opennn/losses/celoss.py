from torch.nn import CrossEntropyLoss
from torch import nn
import torch


def celoss():
    return CrossEntropyLoss()


def customceloss():
    return CustomCrossEntropyLoss()


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
