from .adam import adam, adamw, adamax, radam, nadam


def get_optimizer(name, model, lr=None, betas=None, eps=None, weight_decay=None):
    '''
    Return optimizer by name.

    Parameterts
    -----------
    name : str
        optimizer name ['adam', 'adamw', 'adamax', 'radam', 'nadam'].

    model : Any
        pytorch model.

    lr : float, optional
        learning rate.

    betas : tuple(float, float), optional
        betas.

    eps : float, optional
        eps.

    weight_decay : float, optional
        l2 regularization.
    '''
    if name == 'adam':
        optimizer = adam(model.parameters(), lr, betas, eps, weight_decay)
    elif name == 'adamw':
        optimizer = adamw(model.parameters(), lr, betas, eps, weight_decay)
    elif name == 'adamax':
        optimizer = adamax(model.parameters(), lr, betas, eps, weight_decay)
    elif name == 'radam':
        optimizer = radam(model.parameters(), lr, betas, eps, weight_decay)
    elif name == 'nadam':
        optimizer = nadam(model.parameters(), lr, betas, eps, weight_decay)
    else:
        raise ValueError(f'no optimizer {name}')
    return optimizer
