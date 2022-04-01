import yaml
from algo import train, test, vizualize
from optimizers import adam
from schedulers import steplr
from datasets import mnist, cifar10, cifar100
from losses import celoss, bce, bcelogits, custombce, custombcelogits, customceloss
from losses import mse, mae, custom_mse, custom_mae
from metrics import accuracy, precision, recall, f1_score
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
    metrics_fn = []
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
    optim = config['optimizer']
    sched = config['scheduler']
    step = int(config['step'])
    if 'gamma' in config.keys():
        gamma = float(config['gamma'])
    else:
        gamma = 0.1

    np.random.seed(seed)
    torch.manual_seed(seed)

    if device != 'cpu' and device != 'cuda':
        raise ValueError(f'no device {device}')

    encoder = get_encoder(encoder_name, inc).to(device)
    model = get_decoder(decoder_name, encoder, nc, decoder_mode, device).to(device)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)

    if dataset == 'mnist':
        train_data, valid_data, test_data = mnist(train_part, valid_part, transform)
    elif dataset == 'cifar10':
        train_data, valid_data, test_data = cifar10(train_part, valid_part, transform)
    elif dataset == 'cifar100':
        train_data, valid_data, test_data = cifar100(train_part, valid_part, transform)
    else:
        raise ValueError(f'no dataset {dataset}')

    if optim == 'adam':
        optimizer = adam(model.parameters(), lr, wd)
    else:
        raise ValueError(f'no optimizer {optim}')

    if sched == 'steplr':
        scheduler = steplr(optimizer, step, gamma)
    else:
        raise ValueError(f'no scheduler {sched}')

    one_hot = False
    if loss_fn == 'ce':
        loss_fn = celoss()
    elif loss_fn == 'custom_ce':
        loss_fn = customceloss()
    elif loss_fn == 'bce':
        loss_fn = bce()
    elif loss_fn == 'custom_bce':
        loss_fn = custombce()
    elif loss_fn == 'bce_logits':
        loss_fn = bcelogits()
    elif loss_fn == 'custom_bce_logits':
        loss_fn = custombcelogits()
    elif loss_fn == 'mse':
        loss_fn = mse()
        one_hot = True
    elif loss_fn == 'custom_mse':
        loss_fn = custom_mse()
        one_hot = True
    elif loss_fn == 'mae':
        loss_fn = mae()
        one_hot = True
    elif loss_fn == 'custom_mae':
        loss_fn = custom_mae()
        one_hot = True
    else:
        raise ValueError(f'no loss function {loss_fn}')

    for metric in metrics:
        if metric == 'accuracy':
            acc = accuracy()
            metrics_fn.append(acc)
        elif metric == 'precision':
            prec = precision(nc)
            metrics_fn.append(prec)
        elif metric == 'recall':
            rec = recall(nc)
            metrics_fn.append(rec)
        elif metric == 'f1_score':
            f1 = f1_score(nc)
            metrics_fn.append(f1)
        else:
            raise ValueError(f'no metric {metric}')

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
