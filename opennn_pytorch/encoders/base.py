from .alexnet import AlexnetFeatures
from .lenet import LenetFeatures
from .resnet import Resnet18Features, Resnet34Features, Resnet50Features, Resnet101Features, Resnet152Features
from .googlenet import GoognetNoAuxFeatures
from .mobilenet import MobilenetFeatures
from .vgg import VGG11Features, VGG16Features, VGG19Features


def get_encoder(name, inc):
    '''
    Return encoder object by name.

    Parameterts
    -----------
    name : str
        name of encoder.

    inc : int
        number of input channels [1, 3, 4].
    '''
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
    elif name == 'googlenet':
        return GoognetNoAuxFeatures(inc)
    elif name == 'mobilenet':
        return MobilenetFeatures(inc)
    elif name == 'vgg11':
        return VGG11Features(inc)
    elif name == 'vgg16':
        return VGG16Features(inc)
    elif name == 'vgg19':
        return VGG19Features(inc)
    else:
        raise ValueError(f'no encoder {name}')
