from .alexnet import AlexnetFeatures
from .lenet import LenetFeatures
from .resnet import Resnet18Features, Resnet34Features, Resnet50Features, Resnet101Features, Resnet152Features


def get_encoder(name, inc):
    if name == 'alexnet':
        return AlexnetFeatures(inc)
    elif name == 'lenet':
        return LenetFeatures(inc)
    elif name == 'resnet18':
        return Resnet18Features(inc)
    elif name == 'resnet34':
        return Resnet34Features(inc)
    elif name == 'resnet50':
        return Resnet50Features(inc)
    elif name == 'resnet101':
        return Resnet101Features(inc)
    elif name == 'resnet152':
        return Resnet152Features(inc)
    else:
        raise ValueError(f'no encoder {name}')
