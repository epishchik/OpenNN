from torch.optim import Adam, NAdam, RAdam, AdamW, Adamax


def adam(parameters, lr, betas, eps, weight_decay):
    return Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def nadam(parameters, lr, betas, eps, weight_decay):
    return NAdam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def radam(parameters, lr, betas, eps, weight_decay):
    return RAdam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def adamw(parameters, lr, betas, eps, weight_decay):
    return AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def adamax(parameters, lr, betas, eps, weight_decay):
    return Adamax(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
