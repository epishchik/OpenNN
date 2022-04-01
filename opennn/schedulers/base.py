from .steplr import steplr, multisteplr


def get_scheduler(name, optim, step=None, gamma=None, milestones=None):
    if name == 'steplr':
        scheduler = steplr(optim, step, gamma)
    elif name == 'multisteplr':
        scheduler = multisteplr(optim, milestones, gamma)
    else:
        raise ValueError(f'no scheduler {name}')
    return scheduler
