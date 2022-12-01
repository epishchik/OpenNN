import torchvision.datasets as datasets
from torch.utils.data import random_split
from .custom import CUSTOM


def get_dataset(name,
                tr_part,
                va_part,
                te_part,
                transform,
                datafiles=None):
    if datafiles is None:
        dataset_object = getattr(datasets, name)
        dataset = dataset_object(root='.',
                                 download=True,
                                 transform=transform)
    else:
        dataset = CUSTOM(datafiles[0], datafiles[1], transform)

    tr_part = int(tr_part * len(dataset))
    va_part = int(va_part * len(dataset))
    te_part = int(te_part * len(dataset))

    if tr_part + va_part + te_part != len(dataset):
        sizes = [tr_part,
                 va_part,
                 te_part,
                 len(dataset) - tr_part - va_part - te_part]
    else:
        sizes = [tr_part, va_part, te_part]

    split_dataset = random_split(dataset, sizes)
    train_dataset = split_dataset[0]
    valid_dataset = split_dataset[1]
    test_dataset = split_dataset[2]

    return train_dataset, valid_dataset, test_dataset
