import torch
from torch import nn
from . import decoder


class Single(nn.Module):
    def __init__(self, name, encoder, nc, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.inf = encoder.out_features()

        decoder_object = getattr(decoder, name)
        self.decoder = decoder_object(self.inf, nc).to(device)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


class Multi(nn.Module):
    def __init__(self, name, encoder, nc, device):
        super().__init__()
        self.nc = nc
        self.encoder = encoder
        self.inf = encoder.out_features()
        self.decoders = []

        for i in range(nc):
            decoder_object = getattr(decoder, name[i])
            self.decoders.append(decoder_object(self.inf, 1).to(device))

    def forward(self, x):
        features = self.encoder(x)
        res = torch.cat([self.decoders[i](features)
                        for i in range(self.nc)], dim=1)
        return res
