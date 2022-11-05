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
import wandb

import warnings
warnings.filterwarnings('ignore')


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
    Parse .yaml config and transforms .yaml config
    and generate full train/test pipeline.

    Parameterts
    -----------
    yaml : str
        main .yaml config with all basic parameters.

    Config Attributes
    -----------------
    model : structure
        architecture : structure
            encoder : str
                [lenet, alexnet, googlenet, resnet18, resnet34, resnet50,
                 resnet101, resnet152, mobilenet, vgg11, vgg16, vgg19]

            decoder : str, optional
                [lenet, alexnet, linear]

            multidecoder : list[decoder], optional

        features : structure
            in_channels : int

            number_classes : int

        checkpoint : str, optional
            if specify this checkpoint will be loaded into model.

    algorithm : structure
        name : str
            [train, test]

        device : str
            [cpu, cuda]

        epochs : int

        seed : int

    dataset : structure
        name : str
            [mnist, fashion_mnist, cifar10, cifar100, gtsrb, custom]

        images : str, optional
            specify path to folder with images, only for custom dataset.

        annotation : str, optional
            specify path to yaml file with labels - images specification,
            only for custom dataset.

        batch_size : int

        sizes : structure
            train_size : float
                [0.0 - 1.0]

            valid_size : float
                [0.0 - 1.0]

            test_size : float
                [0.0 - 1.0]

        transform : str
            auxiliary .yaml config with sequential transforms
            for image preprocessing.

    save : structure
        logs : structure
            path : str

        checkpoints : structure
            path : str

            save_every : int

    optimizer : structure
        nane : str
            [adam, adamw, adamax, radam, nadam]

        params : structure
            learning_rate : float

            betas : list[float, float], optional

            eps : float, optional

            weight_decay : float, optional

    scheduler : structure
        name : str
            [steplr, multisteplr, polylr]

        params : structure
            step : int, optional

            gamma : float, optional

            milestones : list[int], optional

            max_decay_steps : int, optional

            end_lr : float, optional

            power : float, optional

    loss_function : str
        [ce, custom_ce, bce, custom_bce, bce_logits, custom_bce_logits, mse,
         custom_mse, mae, custom_mae]

    metrics : list[str]
        str : [accuracy, precision, recall, f1_score]
        len(str) : [1 - 4]

    class_names : list[str], optional
        if specify 10 random images will be vizualized.

    wandb: structure, optional
        project_name : str
            name of wandb project.

        run_name : str
            name of run in wandb project.

        metrics : list[str]
            metric names which will be logged by wandb.

    Transform Config Attributes
    ---------------------------
    tensor : bool

    resize : int

    normalize : list[list[float], list[float]]
        [[means], [stds]] - where means and stds count
        = number of input channels
    '''
    torch.cuda.empty_cache()

    config = parse_yaml(yaml)
    transform_config = parse_yaml(config['dataset']['transform'])
    transform_lst = transforms_lst(transform_config)
    transform = transforms.Compose(transform_lst)

    encoder_name = config['model']['architecture']['encoder']

    if 'decoder' in config['model']['architecture'].keys():
        decoder_name = config['model']['architecture']['decoder']
        decoder_mode = None
    else:
        decoder_name = config['model']['architecture']['multidecoder']
        decoder_mode = 'multidecoder'

    inc = int(config['model']['features']['in_channels'])
    nc = int(config['model']['features']['number_classes'])

    device = config['algorithm']['device']

    algorithm = config['algorithm']['name']

    dataset = config['dataset']['name']
    if dataset == 'custom':
        datafiles = (config['dataset']['images'],
                     config['dataset']['annotation'])
    else:
        datafiles = None

    train_part = float(config['dataset']['sizes']['train_size'])
    valid_part = float(config['dataset']['sizes']['valid_size'])
    test_part = float(config['dataset']['sizes']['test_size'])
    seed = int(config['algorithm']['seed'])

    bs = int(config['dataset']['batch_size'])
    epochs = int(config['algorithm']['epochs'])

    logs = config['save']['logs']['path']
    if not os.path.isdir(logs):
        cwd = os.getcwd().replace('\\', '/')
        os.mkdir(os.path.join(cwd, logs), 0o777)

    loss_fn = config['loss_function']
    metrics = config['metrics']

    checkpoints = config['save']['checkpoints']['path']
    if not os.path.isdir(checkpoints):
        cwd = os.getcwd().replace('\\', '/')
        os.mkdir(os.path.join(cwd, checkpoints), 0o777)

    if algorithm == 'test' and 'class_names' in config.keys():
        names = config['class_names']
        pred = True
    else:
        pred = False

    if 'checkpoint' in config['model'].keys():
        checkpoint = config['model']['checkpoint']
    else:
        checkpoint = None

    se = int(config['save']['checkpoints']['save_every'])
    lr = float(config['optimizer']['params']['learning_rate'])

    if 'weight_decay' in config['optimizer']['params'].keys():
        wd = float(config['optimizer']['params']['weight_decay'])
    else:
        wd = 0.0

    if 'eps' in config['optimizer']['params'].keys():
        opt_eps = float(config['optimizer']['params']['eps'])
    else:
        opt_eps = 1e-8

    if 'betas' in config.keys():
        betas = tuple(list(map(float, config['optimizer']['params']['betas'])))
    else:
        betas = (0.9, 0.999)

    optim = config['optimizer']['name']
    sched = config['scheduler']['name']

    if 'step' in config['scheduler']['params'].keys():
        step = int(config['scheduler']['params']['step'])
    else:
        step = 10

    if 'gamma' in config['scheduler']['params'].keys():
        gamma = float(config['scheduler']['params']['gamma'])
    else:
        gamma = 0.1

    if 'milestones' in config['scheduler']['params'].keys():
        milestones = list(
            map(int, config['scheduler']['params']['milestones']))
    else:
        milestones = [10, 30, 70, 150]

    if 'max_decay_steps' in config['scheduler']['params'].keys():
        mdsteps = int(config['scheduler']['params']['max_decay_steps'])
    else:
        mdsteps = 100

    if 'end_lr' in config.keys():
        end_lr = float(config['scheduler']['params']['end_lr'])
    else:
        end_lr = 0.00001

    if 'power' in config['scheduler']['params'].keys():
        power = float(config['scheduler']['params']['power'])
    else:
        power = 1.0

    wandb_flag = False
    wandb_metrics = None

    if 'wandb' in config.keys():
        wandb_flag = True
        wandb_dct = {}

        wandb_dct['algorithm'] = algorithm
        wandb_dct['seed'] = seed
        wandb_dct['encoder'] = encoder_name
        wandb_dct['decoder'] = decoder_name
        wdm = 'single' if decoder_mode is None else 'multi'
        wandb_dct['decoder mode'] = wdm
        wandb_dct['input channels'] = inc
        wandb_dct['number classes'] = nc
        wandb_dct['device'] = device
        wandb_dct['initial learning rate'] = lr
        wandb_dct['weight decay'] = wd
        wandb_dct['batch size'] = bs
        wandb_dct['dataset name'] = config['dataset']['name']
        wandb_dct['train size'] = train_part
        wandb_dct['valid size'] = valid_part
        wandb_dct['test size'] = test_part
        wandb_dct['loss function'] = config['loss_function']
        wandb_dct['optimizer'] = config['optimizer']['name']
        wandb_dct['scheduler'] = config['scheduler']['name']

        wandb.init(project=config['wandb']['project_name'],
                   name=config['wandb']['run_name'],
                   config=wandb_dct)
        wandb_metrics = config['wandb']['metrics']

    np.random.seed(seed)
    torch.manual_seed(seed)

    if device != 'cpu' and device != 'cuda':
        raise ValueError(f'no device {device}')

    encoder = get_encoder(encoder_name, inc).to(device)
    model = get_decoder(decoder_name,
                        encoder,
                        nc,
                        decoder_mode,
                        device).to(device)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)

    train_data, valid_data, test_data = get_dataset(dataset,
                                                    train_part,
                                                    valid_part,
                                                    test_part,
                                                    transform,
                                                    datafiles)

    optimizer = get_optimizer(optim,
                              model,
                              lr=lr,
                              betas=betas,
                              eps=opt_eps,
                              weight_decay=wd)

    scheduler = get_scheduler(sched,
                              optimizer,
                              step=step,
                              gamma=gamma,
                              milestones=milestones,
                              max_decay_steps=mdsteps,
                              end_lr=end_lr,
                              power=power)

    loss_fn, one_hot = get_loss(loss_fn)
    metrics_fn = get_metrics(metrics, nc=nc)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=bs,
                                                   shuffle=True)

    valid_dataloader = torch.utils.data.DataLoader(valid_data,
                                                   batch_size=bs,
                                                   shuffle=False)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=1,
                                                  shuffle=False)

    if algorithm == 'train':
        train(train_dataloader,
              valid_dataloader,
              model,
              optimizer,
              scheduler,
              loss_fn,
              metrics_fn,
              epochs,
              checkpoints,
              logs,
              device,
              se,
              one_hot,
              nc,
              wandb_flag,
              wandb_metrics)
    elif algorithm == 'test':
        test_logs = test(test_dataloader,
                         model,
                         loss_fn,
                         metrics_fn,
                         logs,
                         device,
                         one_hot,
                         nc,
                         wandb_flag,
                         wandb_metrics)
        if pred:
            indices = random.sample(range(0, len(test_data)), 10)
            os.mkdir(test_logs + '/prediction', 0o777)
            for i in range(10):
                tmp_range = range(nc)
                tmp_dct = {i: names[i] for i in tmp_range}
                prediction(test_data,
                           model,
                           device,
                           tmp_dct,
                           test_logs + f'/prediction/{i}',
                           indices[i])
    else:
        raise ValueError(f'no algorithm {algorithm}')
