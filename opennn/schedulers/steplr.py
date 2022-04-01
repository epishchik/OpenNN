from torch.optim.lr_scheduler import StepLR, MultiStepLR


def steplr(optimizer, step, gamma):
    scheduler = StepLR(optimizer, step, gamma)
    return scheduler


def multisteplr(optimizer, milestones, gamma):
    scheduler = MultiStepLR(optimizer, milestones, gamma)
    return scheduler
