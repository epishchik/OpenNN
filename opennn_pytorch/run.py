from opennn_pytorch.algo import train, test, prediction
from opennn_pytorch.optimizer import get_optimizer
from opennn_pytorch.scheduler import get_scheduler
from opennn_pytorch.dataset import get_dataset
from opennn_pytorch.metric import get_metric
from opennn_pytorch.loss import get_loss
from opennn_pytorch.encoder import get_encoder
from opennn_pytorch.decoder import get_decoder

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
    with open(config, 'r') as stream:
        data = yaml.safe_load(stream)
    return data


def transforms_lst(transform_config):
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
    torch.cuda.empty_cache()

    config = parse_yaml(yaml)

    transform_config = parse_yaml(config['dataset']['transform'])
    transform_lst = transforms_lst(transform_config)
    transform = transforms.Compose(transform_lst)

    encoder_name = config['model']['architecture']['encoder']

    decoder_name = config['model']['architecture']['decoder']
    decoder_mode = 'Single' if isinstance(decoder_name, str) else 'Multi'

    inc = int(config['model']['features']['in_channels'])
    nc = int(config['model']['features']['number_classes'])

    device_str = config['algorithm']['device']
    device = torch.device(device_str)

    algorithm = config['algorithm']['name']

    dataset = config['dataset']['name']
    if dataset == 'CUSTOM':
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

    optimizer_name = config['optimizer']['name']
    optimizer_params = config['optimizer']['params']

    scheduler_name = config['scheduler']['name']
    scheduler_params = config['scheduler']['params']
    scheduler_type = config['scheduler']['type']

    wandb_flag = False
    wandb_metrics = None

    if 'wandb' in config.keys():
        wandb_flag = True
        wandb_dct = {}

        wandb_dct['algorithm'] = algorithm
        wandb_dct['seed'] = seed
        wandb_dct['encoder'] = encoder_name
        wandb_dct['decoder'] = decoder_name
        wandb_dct['decoder mode'] = decoder_mode
        wandb_dct['input channels'] = inc
        wandb_dct['number classes'] = nc
        wandb_dct['device'] = device_str
        wandb_dct['batch size'] = bs
        wandb_dct['dataset name'] = config['dataset']['name']
        wandb_dct['loss function'] = config['loss_function']
        wandb_dct['optimizer'] = config['optimizer']['name']
        wandb_dct['scheduler'] = config['scheduler']['name']
        wandb_dct['initial lr'] = config['optimizer']['params']['lr']

        wandb.init(project=config['wandb']['project_name'],
                   name=config['wandb']['run_name'],
                   config=wandb_dct)
        wandb_metrics = config['wandb']['metrics']

    np.random.seed(seed)
    torch.manual_seed(seed)

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

    optimizer = get_optimizer(optimizer_name, model, optimizer_params)

    scheduler = get_scheduler(scheduler_name,
                              optimizer,
                              scheduler_params,
                              scheduler_type)

    loss_fn, one_hot = get_loss(loss_fn)
    metrics_fn = get_metric(metrics, nc=nc)

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
