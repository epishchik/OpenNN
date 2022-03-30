import torchvision
import torch


def cifar10(tr_part, va_part, transform):
    dataset = torchvision.datasets.CIFAR10('.', download=True, transform=transform)
    tr_part = int(tr_part * len(dataset))
    va_part = int(va_part * len(dataset))
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [tr_part, va_part, len(dataset) - tr_part - va_part])
    return train_dataset, valid_dataset, test_dataset


# TODO проверить cifar100
def cifar100(tr_part, va_part, transform):
    dataset = torchvision.datasets.CIFAR100('.', download=True, transform=transform)
    tr_part = int(tr_part * len(dataset))
    va_part = int(va_part * len(dataset))
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [tr_part, va_part, len(dataset) - tr_part - va_part])
    return train_dataset, valid_dataset, test_dataset
