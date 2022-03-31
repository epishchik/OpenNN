from .model import Model
from .model import MultiDecModel


def get_decoder(name, encoder, nc, mode, device):
    if mode != 'multi_decoders':
        return Model(name, encoder, nc, device)
    else:
        return MultiDecModel(name, encoder, nc, device)
