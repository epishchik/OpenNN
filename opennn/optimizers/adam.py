from torch.optim import Adam


def adam(parameters, lr, weight_decay):
    return Adam(parameters, lr=lr, weight_decay=weight_decay)
