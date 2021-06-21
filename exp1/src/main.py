import torch as t
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from matplotlib import pyplot as plt
from model import FNN
import copy


class FuncFitter(object):
    def __init__(self, to_fit: str or callable, x_range: tuple):
        if isinstance(to_fit, str):
            self.to_fit = eval(f'np.{to_fit}')
        else:
            self.to_fit = to_fit
        self.x_range = x_range

    def gen_data(self, data_size, train_ratio, random_state):
        np.random.seed(random_state)
        x = np.linspace(self.x_range[0], self.x_range[1], data_size)[:, np.newaxis]
        y = self.to_fit(x)
        indices = np.random.permutation(data_size)
        ids_train, ids_test = np.split(indices, [round(train_ratio * data_size)])
        x_train, y_train = x[ids_train], y[ids_train]
        x_test, y_test = x[ids_test], y[ids_test]
        self.train_data = (t.Tensor(x_train), t.Tensor(y_train))
        self.test_data = (t.Tensor(x_test), t.Tensor(y_test))
        return self.train_data, self.test_data

    def train(self, model, optim, criterion, epochs, batch_size, pbar='batch'):
        dataset = TensorDataset(*self.train_data)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)
        pbar_epoch = range(epochs)
        if pbar == 'epoch':
            pbar_epoch = tqdm(pbar_epoch, desc='Epochs', unit='epoch',
                              bar_format='{desc:<7.7}{percentage:3.0f}%|{bar:30}{r_bar}')
        for i in pbar_epoch:
            loss = None
            pbar_batch = dataloader
            if pbar == 'batch':
                pbar_batch = tqdm(pbar_batch, desc=f'[Epoch {i + 1}/{epochs}]', unit='batch',
                                  bar_format='{desc:<15.15}{percentage:3.0f}%|{bar:30}{r_bar}')
            for x, y in pbar_batch:
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if pbar == 'batch':
                    pbar_batch.set_postfix({'train_loss': loss.item()})
            pbar_epoch.set_postfix({'train_loss': loss.item()})
        return model

    def test(self, model, criterion, plot=True):
        x_test, y_test = self.test_data
        y_pred = model(x_test)
        loss = criterion(y_pred, y_test).item()

        if plot:
            fig, ax = plt.subplots()
            ax.set_title('Function Fitter: {}\ntest loss: {}'.format(str(self.to_fit), loss))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(self.x_range)
            ax.scatter(x_test.detach().numpy(), y_test.detach().numpy(), s=5, label='true')
            ax.scatter(x_test.detach().numpy(), y_pred.detach().numpy(), s=5, label='pred')
            ax.legend()
            plt.show()

        return loss


if __name__ == '__main__':
    fitter = FuncFitter(np.sin, (0, 4 * np.pi))
    fitter.gen_data(data_size=10000, train_ratio=0.8, random_state=7)
    num_rounds = 10
    losses = []
    for i in range(num_rounds):
        fnn = FNN(neurons=[1, *[20] * 2, 1], activation='tanh')
        if i == 0:
            print('number of parameters: {}'.format(sum(p.numel() for p in fnn.parameters() if p.requires_grad)))
        optim = Adam(fnn.parameters(), lr=10 ** -3)
        criterion = nn.MSELoss()
        fitter.train(fnn, optim, criterion, epochs=10, batch_size=10, pbar='epoch')
        test_loss = fitter.test(fnn, criterion, plot=True)
        losses.append(round(test_loss, 6))
    print(losses)
