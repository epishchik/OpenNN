from torch import optim


def get_optimizer(name, model, params):
    optimizer_object = getattr(optim, name)
    optimizer_instance = optimizer_object(model.parameters(), **params)
    return optimizer_instance
