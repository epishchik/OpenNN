from . import alexnet, lenet, resnet, googlenet, mobilenet, vgg


def get_encoder(name, inc):
    if 'AlexNet' in name:
        model_object = getattr(alexnet, name)
    elif 'LeNet' in name:
        model_object = getattr(lenet, name)
    elif 'ResNet' in name:
        model_object = getattr(resnet, name)
    elif 'GoogleNet' in name:
        model_object = getattr(googlenet, name)
    elif 'MobileNet' in name:
        model_object = getattr(mobilenet, name)
    elif 'VGG' in name:
        model_object = getattr(vgg, name)

    model_instance = model_object(inc)
    return model_instance
