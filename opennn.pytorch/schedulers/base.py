from .steplr import steplr, multisteplr


def get_scheduler(name, optim, step=None, gamma=None, milestones=None):
    '''
    Return scheduler by name.

    Parameterts
    -----------
    name : str
        scheduler name ['steplr', 'multisteplr']
    optim : Optimizer
        optimizer object.
    step : int, optional
        every step epochs do scheduler step.
    gamma : float, optional
        lr will be multiplied to this value each step epochs.
    milestones : list[int], optional
        checkpoints for lr * gamma.
    '''
    if name == 'steplr':
        scheduler = steplr(optim, step, gamma)
    elif name == 'multisteplr':
        scheduler = multisteplr(optim, milestones, gamma)
    else:
        raise ValueError(f'no scheduler {name}')
    return scheduler
