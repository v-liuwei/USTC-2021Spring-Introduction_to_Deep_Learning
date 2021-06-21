import numpy as np
import torch as t
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import TinyImageNet
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNN
from tqdm import tqdm
import os
from pprint import pprint
import random
from copy import deepcopy
import sys
from matplotlib import pyplot as plt


def set_seed(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.backends.cudnn.deterministic = True


def train(model, train_data, epochs,
          lr_init=1e-3, lr_min=1e-5, lr_decay=1., lr_min_delta=0., lr_patience=1,
          val_data=None, val_min_delta=0., val_patience=1,
          restore_best_weights=True, top=1, verbose=True, device='cpu'):
    min_val_loss = np.inf
    wait = 0
    best_weights = None
    history = {}
    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr_init)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=lr_decay, patience=lr_patience,
                                  threshold=lr_min_delta, verbose=verbose, min_lr=lr_min)
    loss_fc = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        pbar_batch = train_data
        if verbose:
            pbar_batch = tqdm(train_data, desc=f'[Epoch {epoch + 1}/{epochs}]', unit='batch',
                              ascii=True, bar_format='{desc:<13.13}{percentage:3.0f}%|{bar:10}{r_bar}')
        for i, data in enumerate(pbar_batch, 1):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fc(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicts = t.max(outputs, 1)[1]
            batch_total_num = labels.size(0)
            batch_correct_num = (predicts == labels.data).sum().item()
            if verbose:
                pbar_batch.set_postfix({'train_loss': loss.item(),
                                        'train_acc': f'{batch_correct_num}/{batch_total_num}'})
        train_loss, train_acc = validate(model, train_data, 'train', top, verbose, device)

        history.setdefault('epoch', []).append(epoch)
        history.setdefault('train_loss', []).append(train_loss)
        history.setdefault('train_acc', []).append(train_acc[0])
        if val_data is not None:
            val_loss, val_acc = validate(model, val_data, 'val', top, verbose, device)
            history.setdefault('val_loss', []).append(val_loss)
            history.setdefault('val_acc', []).append(val_acc[0])

            # schedule lr
            scheduler.step(val_loss)
            # control early stopping
            if val_loss < min_val_loss - val_min_delta:
                min_val_loss = val_loss
                wait = 0
                best_weights = deepcopy(model.state_dict())
                # print(best_weights['fc.bias'])
            else:
                wait += 1
            if wait > val_patience:
                if verbose:
                    print('>>> Early Stopped.')
                if restore_best_weights:
                    model.load_state_dict(best_weights)
                    # print(model.state_dict().copy()['fc.bias'])
                break
    return model, history


def validate(model, dataloader, mode='val', top=1,  verbose=True, device='cpu'):
    model.eval()
    tops = top if isinstance(top, list) else [top]
    losses = []
    correct_nums = np.zeros_like(tops)
    total_nums = np.zeros_like(tops)
    loss_fc = nn.CrossEntropyLoss()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fc(outputs, labels.long())
        predicts = [t.argsort(outputs, dim=1, descending=True)[:, :top] for top in tops]

        losses.append(loss.item())
        for i, predict in enumerate(predicts):
            total_nums[i] += labels.size(0)
            correct_nums[i] += (predict == labels.data.unsqueeze(1)).sum().item()

    avg_loss = np.mean(losses)
    acc_list = [correct_num / total_num for correct_num, total_num in zip(correct_nums, total_nums)]
    if verbose:
        print(f"{f'{mode}_loss':>11} = {avg_loss:<6.4f}, ", end='')
        print(', '.join([f"{f'{mode}_acc_top{top}':>15} = {acc:<6.4f}" for top, acc in zip(tops, acc_list)]))
    model.train()
    return avg_loss, acc_list


def plot_history(history):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = history['epoch']
    ax1.set_title('Training history')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('acc')
    p1 = ax1.plot(x, history['train_loss'], label='train_loss')
    p2 = ax1.plot(x, history['val_loss'], label='val_loss')
    p3 = ax2.plot(x, history['train_acc'], '-.', label='train_acc')
    p4 = ax2.plot(x, history['val_acc'], '-.', label='val_acc')
    lines = p1 + p2 + p3 + p4
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels)
    plt.savefig('history.png')


def run(params, plot=False, verbose=False):
    cnn = CNN(params['block_sizes'], params['res'], params['norm'],
              params['dropout'][0], params['dropout'][1])
    if verbose:
        print('number of parameters: {}'.format(sum(p.numel() for p in cnn.parameters() if p.requires_grad)))
    cnn, history = train(
        model=cnn,
        train_data=train_dataloader,
        epochs=params['epochs'],
        lr_init=params['lr_init'],
        lr_min=params['lr_min'],
        lr_decay=params['lr_decay'],
        lr_min_delta=params['lr_min_delta'],
        lr_patience=params['lr_patience'],
        val_data=val_dataloader,
        val_min_delta=params['val_min_delta'],
        val_patience=params['val_patience'],
        restore_best_weights=params['restore_best_weights'],
        top=params['top'],
        verbose=verbose,
        device=params['device']
    )
    if plot:
        plot_history(history)
    loss, acc = validate(cnn, val_dataloader, 'val', top=params['top'], verbose=False, device=params['device'])
    return loss, acc


if __name__ == "__main__":
    data_root = '/home/liuwei/projects/DL_exps/exp2/tiny-imagenet-200/'
    batch_size = 256*3

    train_dataset = TinyImageNet(data_root, 'train', transforms.Compose([transforms.ToTensor()]))
    val_dataset = TinyImageNet(data_root, 'val', transforms.Compose([transforms.ToTensor()]))
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)

    default_params = {
        'block_sizes':
            # [
            #     (64, 64, 1),
            #     (64, 128, 2),
            #     (128, 256, 2),
            #     (256, 512, 2),
            #     (512, 1024, 2),
            # ],  # 11 conv layers
            [
                (64, 64, 1),
                (64, 64, 1),
                (64, 64, 1),
                (64, 128, 2),
                (128, 128, 1),
                (128, 128, 1),
                (128, 256, 2),
                (256, 256, 1),
                (256, 256, 1),
                (256, 512, 2),
                (512, 512, 1),
                (512, 512, 1),
                (512, 1024, 2),
                (1024, 1024, 1),
                (1024, 1024, 1),
            ],  # 31 conv layers
        'epochs': 80,
        'res': True,
        'norm': True,
        'dropout': (0.1, 0.5),
        'lr_init': 1e-3,
        'lr_min': 1e-5,
        'lr_decay': 0.5,
        'lr_min_delta': 0.,
        'lr_patience': 2,
        'val_min_delta': 0.,
        'val_patience': 30,
        'top': [1, 5, 10],
        'restore_best_weights': True
    }
    param_grids = {
        'block_sizes': [
            [
                (64, 64, 1),
                (64, 128, 2),
                (128, 256, 2),
                (256, 512, 2),
                (512, 1024, 2),
            ],  # 11 conv layers
            [
                (64, 64, 1),
                (64, 64, 1),
                (64, 128, 2),
                (128, 128, 1),
                (128, 256, 2),
                (256, 256, 1),
                (256, 512, 2),
                (512, 512, 1),
                (512, 1024, 2),
                (1024, 1024, 1)
            ],  # 21 conv layers
            [
                (64, 64, 1),
                (64, 64, 1),
                (64, 64, 1),
                (64, 128, 2),
                (128, 128, 1),
                (128, 128, 1),
                (128, 256, 2),
                (256, 256, 1),
                (256, 256, 1),
                (256, 512, 2),
                (512, 512, 1),
                (512, 512, 1),
                (512, 1024, 2),
                (1024, 1024, 1),
                (1024, 1024, 1),
            ],  # 31 conv layers
        ],
        'res': [True, False],
        'norm': [True, False],
        'dropout': [(0., 0.), (0.1, 0.3), (0.2, 0.5), (0.3, 0.7)],
        'lr_decay': [0.1, 0.5, 0.99],
    }

    set_seed(17717)

    try:
        job = int(sys.argv[1])
    except IndexError as e:
        job = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    device = 'cuda'
    # devices = [0, 1, 2, 3]
    # device = f'cuda:{devices[job]}'

    logfile = f'logs{job}.txt'

    print("default parameters are:")
    pprint(default_params, width=40)
    with open(logfile, 'w') as file:
        file.write("default parameters are:\n")
        pprint(default_params, stream=file, width=40)

    if job == 0:
        f = open(logfile, 'a')
        loss, acc_list = run(default_params, plot=True, verbose=True)
        info = f'val_loss = {loss}, val_acc = {acc_list}'
        print(info)
        f.write(info + '\n')
        f.close()
        exit()

    job_keys = {
        1: ['block_sizes', 'norm'],
        2: ['res'],
        3: ['dropout', 'lr_decay']
    }

    for key in job_keys[job]:
        for v in param_grids[key]:
            if key not in ['block_sizes', 'res'] and v == default_params[key]:
                continue
            f = open(logfile, 'a')
            new_params = default_params.copy()
            if key == 'res':
                new_params.update({'block_sizes': param_grids['block_sizes'][-1]})
                info = f'>>> set `block_sizes` to {param_grids["block_sizes"][-1]}\n'
            else:
                info = ''
            new_params.update({key: v})
            info += f'>>> set `{key}` to {v}'
            print(info)
            f.write(info + '\n')
            loss, acc_list = run(new_params, verbose=True)
            info = f'val_loss = {loss}, val_acc = {acc_list}'
            print(info)
            f.write(info + '\n')
            f.close()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    # run following command in terminal     #
    # $ python main.py 1                    #
    # $ python main.py 2                    #
    # $ python main.py 3                    #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
