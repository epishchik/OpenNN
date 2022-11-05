from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import wandb


def train(train_dataloader,
          valid_dataloader,
          model,
          optimizer,
          scheduler,
          loss_fn,
          metrics,
          epochs,
          checkpoints,
          logs,
          device,
          save_every,
          one_hot,
          nc,
          wandb_flag,
          wandb_metrics):
    '''
    Train pipeline.

    Parameterts
    -----------
    train_dataloader : torch.utils.data.DataLoader
        train dataloader.

    valid_dataloader : torch.utils.data.DataLoader
        valid dataloader.

    model : Any
        pytorch model.

    optimizer : torch.optim.Optimizer
        optimizer for this model.

    scheduler : torch.optim.lr_scheduler
        scheduler for this optimizer.

    loss_fn : Any
        loss function.

    metrics : list[Any]
        list of metric functions.

    epochs : int
        epochs number.

    checkpoints : str
        folder for checkpoints.

    logs : str
        folder for logs.

    device : str
        device ['cpu', 'cuda'].

    save_every : int
        every save_every epoch save model weights.

    one_hot : bool
        one_hot for labels.

    nc : int
        classes number.

    wandb_flag : bool
        use wandb for logging.

    wandb_metrics: list[str]
        metric names which will be logged by wandb.
    '''
    checkpoints_folder = list(map(int, os.listdir(checkpoints)))
    checkpoints_folder = max(checkpoints_folder) + \
        1 if checkpoints_folder != [] else 0
    os.mkdir(f'{checkpoints}/{checkpoints_folder}', mode=0o777)
    checkpoints = f'{checkpoints}/{checkpoints_folder}'

    logs_folder = list(map(int, os.listdir(logs)))
    logs_folder = max(logs_folder) + 1 if logs_folder != [] else 0
    os.mkdir(f'{logs}/{logs_folder}', mode=0o777)
    logs = f'{logs}/{logs_folder}'

    tqdm_iter = tqdm(range(epochs))
    best_loss = 99999999.0
    best_epoch = 0

    range_daigrams = range(2 * (len(metrics) + 1) + 1)
    diagrams0 = [[] for _ in range_daigrams]
    diagrams1 = [[]for _ in range_daigrams]

    diagrams = [diagrams0, diagrams1, []]

    for ne, epoch in enumerate(tqdm_iter):
        model.train()
        train_loss = 0.0
        valid_loss = 0.0
        metric_names = [metric.name() for metric in metrics]
        train_mvct = [0.0] * len(metrics)
        valid_mvct = [0.0] * len(metrics)

        for batch in train_dataloader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            if one_hot:
                labels = F.one_hot(labels, nc).float()

            preds = model(imgs)
            loss = loss_fn(preds, labels)

            for mi, metric in enumerate(metrics):
                train_mvct[mi] += metric.calc(preds, labels).cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)

                if one_hot:
                    labels = F.one_hot(labels, nc).float()

                preds = model(imgs)
                loss = loss_fn(preds, labels)

                for mi, metric in enumerate(metrics):
                    valid_mvct[mi] += metric.calc(preds, labels).cpu()

                valid_loss += loss.item()

        train_mvct = np.array(train_mvct) / len(train_dataloader)
        valid_mvct = np.array(valid_mvct) / len(valid_dataloader)
        train_loss /= len(train_dataloader)
        valid_loss /= len(valid_dataloader)

        if wandb_flag:
            wandb.log({'train loss': train_loss,
                       'valid loss': valid_loss,
                       'learning rate': optimizer.param_groups[0]['lr']})

        if epoch % save_every == 0 or epoch == epochs - 1:
            state_dict = model.state_dict()
            torch.save(state_dict, checkpoints +
                       '/plan_{}_{:.2f}.pt'.format(epoch, valid_loss))

        if valid_loss < best_loss:
            best_epoch = epoch
            best_loss = valid_loss
            state_dict = model.state_dict()
            torch.save(state_dict, checkpoints + '/best.pt')

        tqdm_dct = {}
        train_str = ''
        valid_str = ''

        tqdm_dct['train loss'] = train_loss
        diagrams[1][0].append(train_loss)

        if ne == 0:
            diagrams[2].append('train_loss')

        for i, metric in enumerate(train_mvct):
            if wandb_metrics is not None and metric_names[i] in wandb_metrics:
                wandb.log({f'train_{metric_names[i]}': metric})

            diagrams[1][i + 1].append(metric)
            if ne == 0:
                diagrams[2].append(f'train_{metric_names[i]}')

            tqdm_dct[f'train_{metric_names[i]}'] = metric
            train_str += f'train_{metric_names[i]}: {metric:.3f} '

        tqdm_dct['valid loss'] = valid_loss
        diagrams[1][len(metrics) + 1].append(valid_loss)

        if ne == 0:
            diagrams[2].append('valid_loss')

        for i, metric in enumerate(valid_mvct):
            if wandb_metrics is not None and metric_names[i] in wandb_metrics:
                wandb.log({f'valid_{metric_names[i]}': metric})

            diagrams[1][i + len(metrics) + 2].append(metric)
            if ne == 0:
                diagrams[2].append(f'valid_{metric_names[i]}')

            tqdm_dct[f'valid_{metric_names[i]}'] = metric
            if i != len(valid_mvct) - 1:
                valid_str += f'valid_{metric_names[i]}: {metric:.3f} '
            else:
                valid_str += f'valid_{metric_names[i]}: {metric:.3f}\n'

        tqdm_iter.set_postfix(tqdm_dct, refresh=True)

        with open(logs + '/trainval.log', 'a') as in_f:
            epoch_log = f'epoch: {epoch + 1}/{epochs} '
            loss_log = f'train loss: {train_loss:.3f} '
            loss_log += f'valid loss: {valid_loss:.3f} '
            metric_log = train_str + valid_str
            in_f.write(epoch_log + loss_log + metric_log)

        diagrams[1][-1].append(optimizer.param_groups[0]['lr'])
        for k in range(len(diagrams[0])):
            diagrams[0][k].append(ne)
        if ne == 0:
            diagrams[2].append('lr')

        scheduler.step()
        tqdm_iter.refresh()

    for ind, name in enumerate(diagrams[2]):
        plt.plot(diagrams[0][ind], diagrams[1][ind])
        plt.savefig(logs + f'/{name}.png')
        plt.close()

    os.rename(checkpoints + '/best.pt', checkpoints +
              '/best_{}_{:.2f}.pt'.format(best_epoch, best_loss))


def test(test_dataloader,
         model,
         loss_fn,
         metrics,
         logs,
         device,
         one_hot,
         nc,
         wandb_flag,
         wandb_metrics):
    '''
    Test pipeline.

    Parameterts
    -----------
    test_dataloader : torch.utils.data.DataLoader
        test dataloader.

    model : Any
        pytorch model.

    loss_fn : Any
        loss function.

    metrics : list[Any]
        list of metric functions.

    logs : str
        folder for logs.

    device : str
        device ['cpu', 'cuda'].

    one_hot : bool
        one_hot for labels.

    nc : int
        classes number.

    wandb_flag : bool
        use wandb for logging.

    wandb_metrics: list[str]
        metric names which will be logged by wandb.
    '''
    logs_folder = list(map(int, os.listdir(logs)))
    logs_folder = max(logs_folder) + 1 if logs_folder != [] else 0
    os.mkdir(f'{logs}/{logs_folder}', mode=0o777)
    logs = f'{logs}/{logs_folder}'

    model.eval()
    test_loss = 0.0
    metric_names = [metric.name() for metric in metrics]
    test_mvct = [0.0] * len(metrics)

    with torch.no_grad():
        for batch in test_dataloader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            if one_hot:
                labels = F.one_hot(labels, nc).float()

            preds = model(imgs)
            loss = loss_fn(preds, labels)

            for mi, metric in enumerate(metrics):
                test_mvct[mi] += metric.calc(preds, labels).cpu()

            test_loss += loss.item()

    test_mvct = np.array(test_mvct) / len(test_dataloader)
    test_loss /= len(test_dataloader)
    test_str = ''

    if wandb_flag:
        wandb.log({'test loss': test_loss})

    for i, metric in enumerate(test_mvct):
        if wandb_metrics is not None and metric_names[i] in wandb_metrics:
            wandb.log({f'test_{metric_names[i]}': metric})

        if i != len(test_mvct) - 1:
            test_str += f'{metric_names[i]}: {metric:.3f} '
        else:
            test_str += f'{metric_names[i]}: {metric:.3f}\n'

    with open(logs + '/test.log', 'a') as in_f:
        in_f.write(f'test loss: {test_loss:.3f} ' + test_str)

    return logs
