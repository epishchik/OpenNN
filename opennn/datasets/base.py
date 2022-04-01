from .mnist import mnist
from .cifar import cifar10, cifar100


def get_dataset(name, train_part, valid_part, transform):
    if name == 'mnist':
        train_data, valid_data, test_data = mnist(train_part, valid_part, transform)
    elif name == 'cifar10':
        train_data, valid_data, test_data = cifar10(train_part, valid_part, transform)
    elif name == 'cifar100':
        train_data, valid_data, test_data = cifar100(train_part, valid_part, transform)
    else:
        raise ValueError(f'no dataset {name}')
    return train_data, valid_data, test_data
