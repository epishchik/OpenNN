import yaml
from algo import train, test, vizualize
from optimizers import get_optimizer
from schedulers import get_scheduler
from datasets import get_dataset
from metrics import get_metrics
from losses import get_loss
from encoders import get_encoder
from decoders import get_decoder
import numpy as np
import torch
from torchvision import transforms


def parse_yaml(config):
    with open(config, 'r') as stream:
        data = yaml.safe_load(stream)
    return data


def transforms_lst(transform_config):
    lst = []
    for el in transform_config.keys():
        if el == 'tensor' and transform_config['tensor'] == 'yes':
            lst.append(transforms.ToTensor())
        elif el == 'resize':
            size = int(transform_config['resize'])
            lst.append(transforms.Resize((size, size)))
        else:
            raise ValueError(f'no augmentation {el}: {transform_config[el]}')
    return lst


def run(yaml, transform_yaml):
    torch.cuda.empty_cache()

    config = parse_yaml(yaml)
    transform_config = parse_yaml(transform_yaml)
    transform_lst = transforms_lst(transform_config)
    transform = transforms.Compose(transform_lst)

    encoder_name = config['encoder']
    if 'decoder' in config.keys():
        decoder_name = config['decoder']
        decoder_mode = None
    else:
        decoder_name = config['multidecoder']
        decoder_mode = 'multi_decoders'
    inc = int(config['in_channels'])
    nc = int(config['number_classes'])
    device = config['device']
    algorithm = config['algorithm']
    dataset = config['dataset']
    train_part = float(config['train_part'])
    valid_part = float(config['valid_part'])
    seed = int(config['seed'])
    bs = int(config['batch_size'])
    epochs = int(config['epochs'])
    logs = config['logs']
    loss_fn = config['loss']
    metrics = config['metrics']
    checkpoints = config['checkpoints']
    if algorithm == 'test' and 'class_names' in config.keys():
        names = config['class_names']
        viz = True
    else:
        viz = False
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

    np.random.seed(seed)
    torch.manual_seed(seed)

    if device != 'cpu' and device != 'cuda':
        raise ValueError(f'no device {device}')

    encoder = get_encoder(encoder_name, inc).to(device)
    model = get_decoder(decoder_name, encoder, nc, decoder_mode, device).to(device)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)

    train_data, valid_data, test_data = get_dataset(dataset, train_part, valid_part, transform)
    optimizer = get_optimizer(optim, model, lr=lr, betas=betas, eps=opt_eps, weight_decay=wd)
    scheduler = get_scheduler(sched, optimizer, step=step, gamma=gamma, milestones=milestones)
    loss_fn, one_hot = get_loss(loss_fn)
    metrics_fn = get_metrics(metrics, nc=nc)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=bs, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    if algorithm == 'train':
        train(train_dataloader, valid_dataloader, model, optimizer, scheduler, loss_fn, metrics_fn, epochs, checkpoints, logs, device, se, one_hot, nc)
    elif algorithm == 'test':
        test(test_dataloader, model, loss_fn, metrics_fn, logs, device, one_hot, nc)
        if viz:
            for _ in range(10):
                vizualize(valid_data, model, device, {i: names[i] for i in range(nc)})
    else:
        raise ValueError(f'no algorithm {algorithm}')


if __name__ == '__main__':
    run('C:/Users/SuperPC/Downloads/OpenNN/experiments/config.yaml', 'C:/Users/SuperPC/Downloads/OpenNN/experiments/transform.yaml')
