from torch.optim.lr_scheduler import StepLR


def steplr(optimizer, step, gamma):
    scheduler = StepLR(optimizer, step, gamma)
    return scheduler
