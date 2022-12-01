from . import custom
from torch import nn


def _is_one_hot(name):
    cond1 = name != 'CustomCrossEntropyLoss'
    cond2 = name != 'CrossEntropyLoss'

    return cond1 and cond2


def get_loss(loss_fn):
    if 'Custom' in loss_fn:
        loss_object = getattr(custom, loss_fn)
    else:
        loss_object = getattr(nn, loss_fn)

    loss_instance = loss_object()
    one_hot = _is_one_hot(loss_fn)

    return loss_instance, one_hot
