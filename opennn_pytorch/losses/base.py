from .celoss import celoss, customceloss
from .bceloss import bce, bcelogits, custombce, custombcelogits
from .meanloss import mse, mae, custom_mse, custom_mae


def get_loss(loss_fn):
    '''
    Return loss_fn object by name and one_hot flag.

    Parameterts
    -----------
    loss_fn : str
        name of loss function.
    '''
    one_hot = False
    if loss_fn == 'ce':
        loss_fn = celoss()
    elif loss_fn == 'custom_ce':
        loss_fn = customceloss()
    elif loss_fn == 'bce':
        loss_fn = bce()
        one_hot = True
    elif loss_fn == 'custom_bce':
        loss_fn = custombce()
        one_hot = True
    elif loss_fn == 'bce_logits':
        loss_fn = bcelogits()
        one_hot = True
    elif loss_fn == 'custom_bce_logits':
        loss_fn = custombcelogits()
        one_hot = True
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
