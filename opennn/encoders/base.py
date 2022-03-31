from .alexnet import AlexnetFeatures
from .lenet import LenetFeatures


def get_encoder(name, inc):
    if name == 'alexnet':
        return AlexnetFeatures(inc)
    elif name == 'lenet':
        return LenetFeatures(inc)
    else:
        raise ValueError(f'no encoder {name}')
