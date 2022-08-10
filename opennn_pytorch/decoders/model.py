import torch
from torch import nn
from .alexnet import AlexnetDecoder
from .lenet import LenetDecoder
from .linear import LinearDecoder


class Model(nn.Module):
    '''
    Class used to combine encoder and decoder models. One encoder - one decoder.

    Attributes
    ----------
    encoder : Any
        encoder model.

    decoder : Any
        decoder model.

    inf : int
        number of encoder output features, used as number input features for decoder.

    Methods
    -------
    forward(x)
        encode input image x into features, decode features into output.
    '''
    def __init__(self, name, encoder, nc, device):
        '''
        Create model by combine one encoder and one decoder.

        Parameters
        ----------
        name : str
            decoder name.

        encoder : Any
            encoder model.

        nc : int
            classes number.

        device : str
            device ['cpu', 'cuda']
        '''
        super().__init__()
        self.encoder = encoder.to(device)
        self.inf = encoder.out_features()
        if name == 'alexnet':
            self.decoder = AlexnetDecoder(self.inf, nc).to(device)
        elif name == 'lenet':
            self.decoder = LenetDecoder(self.inf, nc).to(device)
        elif name == 'linear':
            self.decoder = LinearDecoder(self.inf, nc).to(device)
        else:
            raise ValueError(f'no decoder {name}')

    def forward(self, x):
        '''
        Encode input image x into features, decode features into output.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        features = self.encoder(x)
        return self.decoder(features)


class MultiDecModel(nn.Module):
    '''
    Class used to combine encoder and decoder models. One encoder - nc decoders.

    Attributes
    ----------
    encoder : Any
        encoder model.

    decoders : list[Any]
        list with decoder models.

    inf : int
        number of encoder output features, used as number input features for decoder.

    nc : int
        classes number.

    Methods
    -------
    forward(x)
        encode input image x into features, decode features into output.
    '''
    def __init__(self, name, encoder, nc, device):
        '''
        Create model by combine one encoder and nc decoders.

        Parameters
        ----------
        name : str
            decoder name.

        encoder : Any
            encoder model.

        nc : int
            classes number.

        device : str
            device ['cpu', 'cuda']
        '''
        super().__init__()
        self.nc = nc
        self.encoder = encoder
        self.inf = encoder.out_features()
        self.decoders = []
        for i in range(nc):
            if name[i] == 'alexnet':
                self.decoders.append(AlexnetDecoder(self.inf, 1).to(device))
            elif name[i] == 'lenet':
                self.decoders.append(LenetDecoder(self.inf, 1).to(device))
            elif name == 'linear':
                self.decoders.append(LinearDecoder(self.inf, 1).to(device))
            else:
                raise ValueError(f'no decoder {name}')

    def forward(self, x):
        '''
        Encode input image x into features, decode features into output.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        features = self.encoder(x)
        res = torch.cat([self.decoders[i](features) for i in range(self.nc)], dim=1)
        return res
