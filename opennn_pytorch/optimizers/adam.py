from torch.optim import Adam, NAdam, RAdam, AdamW, Adamax


def adam(parameters, lr, betas, eps, weight_decay):
    '''
    Return Adam optimizer object.
    https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam

    Parameterts
    -----------
    parameters : Any
        model.parameters()

    lr : float
        learning rate

    betas : tuple(float, float)
        betas

    eps : float
        eps

    weight_decay : float
        l2 regularization
    '''
    return Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def nadam(parameters, lr, betas, eps, weight_decay):
    '''
    Return NAdam optimizer object.
    https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam

    Parameterts
    -----------
    parameters : Any
        model.parameters().

    lr : float
        learning rate.

    betas : tuple(float, float)
        betas.

    eps : float
        eps.

    weight_decay : float
        l2 regularization.
    '''
    return NAdam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def radam(parameters, lr, betas, eps, weight_decay):
    '''
    Return RAdam optimizer object.
    https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam

    Parameterts
    -----------
    parameters : Any
        model.parameters().

    lr : float
        learning rate.

    betas : tuple(float, float)
        betas.

    eps : float
        eps.

    weight_decay : float
        l2 regularization.
    '''
    return RAdam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def adamw(parameters, lr, betas, eps, weight_decay):
    '''
    Return AdamW optimizer object.
    https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW

    Parameterts
    -----------
    parameters : Any
        model.parameters().

    lr : float
        learning rate.

    betas : tuple(float, float)
        betas.

    eps : float
        eps.

    weight_decay : float
        l2 regularization.
    '''
    return AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def adamax(parameters, lr, betas, eps, weight_decay):
    '''
    Return Adamax optimizer object.
    https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax

    Parameterts
    -----------
    parameters : Any
        model.parameters().

    lr : float
        learning rate.

    betas : tuple(float, float)
        betas.

    eps : float
        eps.

    weight_decay : float
        l2 regularization.
    '''
    return Adamax(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
