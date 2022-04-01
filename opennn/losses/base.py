from .celoss import celoss, customceloss
from .bceloss import bce, bcelogits, custombce, custombcelogits
from .meanloss import mse, mae, custom_mse, custom_mae


def get_loss(loss_fn):
    one_hot = False
    if loss_fn == 'ce':
        loss_fn = celoss()
    elif loss_fn == 'custom_ce':
        loss_fn = customceloss()
    elif loss_fn == 'bce':
        loss_fn = bce()
    elif loss_fn == 'custom_bce':
        loss_fn = custombce()
    elif loss_fn == 'bce_logits':
        loss_fn = bcelogits()
    elif loss_fn == 'custom_bce_logits':
        loss_fn = custombcelogits()
    elif loss_fn == 'mse':
        loss_fn = mse()
        one_hot = True
    elif loss_fn == 'custom_mse':
        loss_fn = custom_mse()
        one_hot = True
    elif loss_fn == 'mae':
        loss_fn = mae()
        one_hot = True
    elif loss_fn == 'custom_mae':
        loss_fn = custom_mae()
        one_hot = True
    else:
        raise ValueError(f'no loss function {loss_fn}')
    return loss_fn, one_hot
