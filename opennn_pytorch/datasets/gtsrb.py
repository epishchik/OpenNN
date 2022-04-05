import torchvision
import torch


def gtsrb(tr_part, va_part, transform):
    '''
    Return splited into train, val, test parts gtsrb dataset.

    Parameterts
    -----------
    tr_part : float
        percent of train data.

    va_part : float
        percent of valid data.

    transform : torchvision.transforms
        torchvision transforms object with augmentations.
    '''
    dataset = torchvision.datasets.GTSRB('.', download=True, transform=transform)
    tr_part = int(tr_part * len(dataset))
    va_part = int(va_part * len(dataset))
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [tr_part, va_part, len(dataset) - tr_part - va_part])
    return train_dataset, valid_dataset, test_dataset
