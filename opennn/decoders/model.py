import torch
from torch import nn
from .alexnet import AlexnetDecoder
from .lenet import LenetDecoder


class Model(nn.Module):
    def __init__(self, name, encoder, nc, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.inf = encoder.out_features()
        if name == 'alexnet':
            self.decoder = AlexnetDecoder(self.inf, nc).to(device)
        elif name == 'lenet':
            self.decoder = LenetDecoder(self.inf, nc).to(device)
        else:
            raise ValueError(f'no decoder {name}')

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


class MultiDecModel(nn.Module):
    def __init__(self, name, encoder, nc, device):
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
            else:
                raise ValueError(f'no decoder {name}')

    def forward(self, x):
        features = self.encoder(x)
        res = torch.cat([self.decoders[i](features) for i in range(self.nc)], dim=1)
        return res
