from torch.nn import MSELoss, L1Loss


def mse():
    return MSELoss()


def mae():
    return L1Loss()
