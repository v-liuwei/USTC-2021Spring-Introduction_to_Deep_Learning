from utils import set_seed
from config import Config
import torch
import torch.nn as nn
from torch.optim import Adam
from data import NodeClsDataset
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
from model import GCN
from itertools import product
import pandas as pd


def train(model, data, optimizer, loss_fc):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = loss_fc(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Get the predictions
    preds = torch.argmax(logits, dim=1).flatten()
    acc = (preds[data.train_mask] == data.y[data.train_mask]).cpu().numpy().mean()

    return loss, acc


def evaluate(model, data, loss_fc, mode='val'):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        mask = getattr(data, f'{mode}_mask')
        loss = loss_fc(logits[mask], data.y[mask])
        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        acc = (preds[mask] == data.y[mask]).cpu().numpy().mean()

    return loss, acc


def main(cfg: Config):
    set_seed(cfg.seed)
    dataset = NodeClsDataset(cfg.data_root, cfg.data_name, cfg.num_train_per_class,
                             cfg.num_val, cfg.num_test, transform=NormalizeFeatures())
    # from torch_geometric.datasets import Planetoid
    # dataset = Planetoid(root='./tmp/Cora', name='Cora', split='random', transform=NormalizeFeatures())

    model = GCN(dataset.num_node_features, cfg.hidden_dim, dataset.num_classes,
                n_layers=cfg.n_layers, act=cfg.activations, add_self_loops=cfg.add_self_loop,
                pair_norm=cfg.pair_norm, dropout=cfg.dropout, drop_edge=cfg.drop_edge)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    data = dataset[0].to(cfg.device)
    model = model.to(device=cfg.device)
    criterion = criterion.to(cfg.device)
    if not cfg.test_only:
        best_valid_loss = np.inf
        wait = 0
        for epoch in range(cfg.epochs):
            print(f">>> Epoch {epoch + 1}/{cfg.epochs}")

            train_loss, train_acc = train(model, data, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, data, criterion, mode='val')

            print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc * 100:.2f}%')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                wait = 0
                torch.save(model.state_dict(), './checkpoint/best_weights.pt')
            else:
                wait += 1
                if wait > cfg.patience:
                    print('>>> Early stopped.')
                    break

    print(">>> Testing...")
    model.load_state_dict(torch.load("./checkpoint/best_weights.pt"))
    test_loss, test_acc = evaluate(model, data, criterion, mode='test')
    print(f'\tTest Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%')
    return test_acc


if __name__ == '__main__':
    config = Config()
    # main(config)
    # exit()

    cfg_grid = {
        'data_name': ['citeseer', 'cora'],
        'add_self_loop': [True, False],
        'n_layers': [1, 2, 3, 5, 10],
        'drop_edge': [0, .1, .2, .3, .5],
        'pair_norm': [True, False],
        'activations': ['relu', 'tanh', 'sigmoid']
    }
    results = []
    keys = cfg_grid.keys()
    for values in product(*cfg_grid.values()):
        new_cfg = dict(zip(keys, values))
        config.update(new_cfg)
        acc = main(config)
        results.append([*new_cfg.values, acc])
    df = pd.DataFrame(results, columns=[*cfg_grid.keys(), 'test_acc'])
    df.to_csv('./result.csv', index=False)
