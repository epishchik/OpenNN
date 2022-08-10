from .model import Model
from .model import MultiDecModel


def get_decoder(name, encoder, nc, mode, device):
    '''
    Return model object by name of decoder and encoder.

    Parameterts
    -----------
    name : str
        name of decoder.

    encoder : Any
        encoder model.

    nc : int
        classes number.

    mode : str
        mode of decoder ['decoder', 'multidecoder']

    device : str
        name of device ['cpu', 'cuda']
    '''
    if mode != 'multidecoder':
        return Model(name, encoder, nc, device)
    else:
        return MultiDecModel(name, encoder, nc, device)
