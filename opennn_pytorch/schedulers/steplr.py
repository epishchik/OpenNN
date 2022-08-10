from torch.optim.lr_scheduler import StepLR, MultiStepLR


def steplr(optimizer, step, gamma):
    '''
    Return StepLR scheduler object.
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html

    Parameterts
    -----------
    optimizer : Optimizer
        optimizer object.

    step : int
        every step epochs do scheduler step.

    gamma : float
        lr will be multiplied to this value each step epochs.
    '''
    scheduler = StepLR(optimizer, step, gamma)
    return scheduler


def multisteplr(optimizer, milestones, gamma):
    '''
    Return MultiStepLR scheduler object.
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html

    Parameterts
    -----------
    optimizer : Optimizer
        optimizer object.

    milestones : list[int]
        checkpoints for lr * gamma.

    gamma : float
        lr will be multiplied to this value each step epochs.
    '''
    scheduler = MultiStepLR(optimizer, milestones, gamma)
    return scheduler
