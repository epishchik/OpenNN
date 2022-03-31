from torch.nn import BCELoss, BCEWithLogitsLoss


def bce():
    return BCELoss()


def bcelogits():
    return BCEWithLogitsLoss()
