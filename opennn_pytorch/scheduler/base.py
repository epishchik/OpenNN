from torch.optim import lr_scheduler
from . import custom


def get_scheduler(name, optim, params, scheduler_type):
    if scheduler_type == 'custom':
        scheduler_object = getattr(custom, name)
    else:
        scheduler_object = getattr(lr_scheduler, name)

    scheduler_instance = scheduler_object(optim, **params)
    return scheduler_instance
