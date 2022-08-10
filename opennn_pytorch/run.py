from opennn_pytorch.algo import train, test, prediction
from opennn_pytorch.optimizers import get_optimizer
from opennn_pytorch.schedulers import get_scheduler
from opennn_pytorch.datasets import get_dataset
from opennn_pytorch.metrics import get_metrics
from opennn_pytorch.losses import get_loss
from opennn_pytorch.encoders import get_encoder
from opennn_pytorch.decoders import get_decoder

import yaml
import random
import numpy as np
import torch
from torchvision import transforms
import os


def parse_yaml(config):
    '''
    Parameterts
    -----------
    config : str
        path to .yaml config file.
    '''
    with open(config, 'r') as stream:
        data = yaml.safe_load(stream)
    return data


def transforms_lst(transform_config):
    '''
    Transforms dict into list of torchvision.transforms.

    Parameterts
    -----------
    transform_config : dict[str, Any]
        dict from .yaml config file.
    '''
    lst = []
    for el in transform_config.keys():
        if el == 'tensor' and transform_config['tensor']:
            lst.append(transforms.ToTensor())
        elif el == 'resize':
            size = int(transform_config[el])
            lst.append(transforms.Resize((size, size)))
        elif el == 'normalize':
            means_stds = transform_config[el]
            means = list(map(float, means_stds[0]))
            stds = list(map(float, means_stds[1]))
            lst.append(transforms.Normalize(means, stds))
        else:
            raise ValueError(f'no augmentation {el}: {transform_config[el]}')
    return lst


def run(yaml):
    '''
    Parse .yaml config and transforms .yaml config and generate full train/test pipeline.

    Parameterts
    -----------
    yaml : str
        main .yaml config with all basic parameters.

    Config Attributes
    -----------------
    encoder : str
        [lenet, alexnet, googlenet, resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet, vgg11, vgg16, vgg19]

    decoder : str, optional
        [lenet, alexnet, linear]

    multidecoder : list[decoder], optional

    algorithm : str
        [train, test]

    device : str
        [cpu, cuda]

    in_channels : int

    number_classes : int

    dataset : str
        [mnist, fashion_mnist, cifar10, cifar100, gtsrb, custom]

    images : str
        specify path to folder with images, only for custom dataset.

    annotation : str
        specify path to yaml file with labels - images specification, only for custom dataset.

    train_part : float
        [0.0 - 1.0]

    valid_part : float
        [0.0 - 1.0]

    seed : int

    batch_size : int

    epochs : int

    logs : str

    checkpoints : str

    save_every : int

    optimizer : str
        [adam, adamw, adamax, radam, nadam]

    learning_rate : float

    optimizer_betas : list[float, float], optional

    optimizer_eps : float, optional

    weight_decay : float, optional

    scheduler : str
        [steplr, multisteplr, polylr]

    step : int, optional

    gamma : float, optional

    milestones : list[int], optional

    max_decay_steps : int, optional

    end_lr : float, optional

    power : float, optional

    metrics : list[str]
        str : [accuracy, precision, recall, f1_score]
        len(str) : [1 - 4]

    loss : str
        [ce, custom_ce, bce, custom_bce, bce_logits, custom_bce_logits, mse, custom_mse, mae, custom_mae]

    class_names : list[str], optional
        if specify 10 random images will be vizualized.

    checkpoint : str
        if specify this checkpoint will be loaded into model.

    transform : str
        auxiliary .yaml config with sequential transforms for image preprocessing.

    Transform Config Attributes
    ---------------------------
    tensor : bool

    resize : int

    normalize : list[list[float], list[float]]
        [[means], [stds]] - where means and stds count = number of input channels
    '''
    torch.cuda.empty_cache()

    config = parse_yaml(yaml)
    transform_config = parse_yaml(config['transform'])
    transform_lst = transforms_lst(transform_config)
    transform = transforms.Compose(transform_lst)

    encoder_name = config['encoder']
    if 'decoder' in config.keys():
        decoder_name = config['decoder']
        decoder_mode = None
    else:
        decoder_name = config['multidecoder']
        decoder_mode = 'multidecoder'
    inc = int(config['in_channels'])
    nc = int(config['number_classes'])
    device = config['device']
    algorithm = config['algorithm']
    dataset = config['dataset']
    if dataset == 'custom':
        datafiles = (config['images'], config['annotation'])
    else:
        datafiles = None
    train_part = float(config['train_part'])
    valid_part = float(config['valid_part'])
    test_part = float(config['test_part'])
    seed = int(config['seed'])
    bs = int(config['batch_size'])
    epochs = int(config['epochs'])
    logs = config['logs']
    if not os.path.isdir(logs):
        cwd = os.getcwd().replace('\\', '/')
        os.mkdir(os.path.join(cwd, logs), 0o777)
    loss_fn = config['loss']
    metrics = config['metrics']
    checkpoints = config['checkpoints']
    if not os.path.isdir(checkpoints):
        cwd = os.getcwd().replace('\\', '/')
        os.mkdir(os.path.join(cwd, checkpoints), 0o777)
    if algorithm == 'test' and 'class_names' in config.keys():
        names = config['class_names']
        pred = True
    else:
        pred = False
    if 'checkpoint' in config.keys():
        checkpoint = config['checkpoint']
    else:
        checkpoint = None
    se = int(config['save_every'])
    lr = float(config['learning_rate'])
    if 'weight_decay' in config.keys():
        wd = float(config['weight_decay'])
    else:
        wd = 0.0
    if 'optimizer_eps' in config.keys():
        opt_eps = float(config['optimizer_eps'])
    else:
        opt_eps = 1e-8
    if 'optimizer_betas' in config.keys():
        betas = tuple(list(map(float, config['optimizer_betas'])))
    else:
        betas = (0.9, 0.999)
    optim = config['optimizer']
    sched = config['scheduler']
    if 'step' in config.keys():
        step = int(config['step'])
    else:
        step = 10
    if 'gamma' in config.keys():
        gamma = float(config['gamma'])
    else:
        gamma = 0.1
    if 'milestones' in config.keys():
        milestones = list(map(int, config['milestones']))
    else:
        milestones = [10, 30, 70, 150]
    if 'max_decay_steps' in config.keys():
        mdsteps = int(config['max_decay_steps'])
    else:
        mdsteps = 100
    if 'end_lr' in config.keys():
        end_lr = float(config['end_lr'])
    else:
        end_lr = 0.00001
    if 'power' in config.keys():
        power = float(config['power'])
    else:
        power = 1.0

    np.random.seed(seed)
    torch.manual_seed(seed)

    if device != 'cpu' and device != 'cuda':
        raise ValueError(f'no device {device}')

    encoder = get_encoder(encoder_name, inc).to(device)
    model = get_decoder(decoder_name, encoder, nc, decoder_mode, device).to(device)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)

    train_data, valid_data, test_data = get_dataset(dataset, train_part, valid_part, test_part, transform, datafiles)
    optimizer = get_optimizer(optim, model, lr=lr, betas=betas, eps=opt_eps, weight_decay=wd)
    scheduler = get_scheduler(sched, optimizer, step=step, gamma=gamma, milestones=milestones, max_decay_steps=mdsteps, end_lr=end_lr, power=power)
    loss_fn, one_hot = get_loss(loss_fn)
    metrics_fn = get_metrics(metrics, nc=nc)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=bs, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    if algorithm == 'train':
        train(train_dataloader, valid_dataloader, model, optimizer, scheduler, loss_fn, metrics_fn, epochs, checkpoints, logs, device, se, one_hot, nc)
    elif algorithm == 'test':
        test_logs = test(test_dataloader, model, loss_fn, metrics_fn, logs, device, one_hot, nc)
        if pred:
            indices = random.sample(range(0, len(test_data)), 10)
            os.mkdir(test_logs + '/prediction', 0o777)
            for i in range(10):
                prediction(test_data, model, device, {i: names[i] for i in range(nc)}, test_logs + f'/prediction/{i}', indices[i])
    else:
        raise ValueError(f'no algorithm {algorithm}')
