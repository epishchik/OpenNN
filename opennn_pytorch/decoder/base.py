from . import model


def get_decoder(name, encoder, nc, mode, device):
    model_object = getattr(model, mode)
    model_instance = model_object(name, encoder, nc, device)
    return model_instance
