from .mnist import mnist, fashion_mnist
from .cifar import cifar10, cifar100
from .custom import custom


def get_dataset(name, train_part, valid_part, transform, datafiles=None):
    '''
    Return splited into train, val, test parts dataset by name.

    Parameterts
    -----------
    name : str
        name of dataset.

    train_part : float
        percent of train data.

    valid_part : float
        percent of valid data.

    transform : torchvision.transforms
        torchvision transforms object with augmentations.
    '''
    if name == 'mnist':
        train_data, valid_data, test_data = mnist(train_part, valid_part, transform)
    elif name == 'fashion_mnist':
        train_data, valid_data, test_data = fashion_mnist(train_part, valid_part, transform)
    elif name == 'cifar10':
        train_data, valid_data, test_data = cifar10(train_part, valid_part, transform)
    elif name == 'cifar100':
        train_data, valid_data, test_data = cifar100(train_part, valid_part, transform)
    elif name == 'custom':
        train_data, valid_data, test_data = custom(datafiles[0], datafiles[1], train_part, valid_part, transform)
    else:
        raise ValueError(f'no dataset {name}')
    return train_data, valid_data, test_data
