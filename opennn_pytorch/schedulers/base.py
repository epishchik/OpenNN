from .steplr import steplr, multisteplr
from .custom import polylr


def get_scheduler(name,
                  optim,
                  step=None,
                  gamma=None,
                  milestones=None,
                  max_decay_steps=None,
                  end_lr=None,
                  power=None):
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

    max_decay_steps : int
        maximum number of decay steps.

    end_lr : float
        minimal lr.

    power : float
        power value.
    '''
    if name == 'steplr':
        scheduler = steplr(optim, step, gamma)
    elif name == 'multisteplr':
        scheduler = multisteplr(optim, milestones, gamma)
    elif name == 'polylr':
        scheduler = polylr(optim, max_decay_steps, end_lr, power)
    else:
        raise ValueError(f'no scheduler {name}')
    return scheduler
