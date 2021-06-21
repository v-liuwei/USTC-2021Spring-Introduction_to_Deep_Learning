import torch as t
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, neurons: list, activation: str):
        super(FNN, self).__init__()
        activation = activation.lower()
        act_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'elu': nn.ELU,
            'leakyrelu': nn.LeakyReLU,
            'prelu': nn.PReLU,
            'softplus': nn.Softplus,
        }
        self.fc_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        num_layers = len(neurons) - 1
        for i in range(num_layers):
            self.fc_layers.append(nn.Linear(neurons[i], neurons[i+1]))
            if i < num_layers - 1:
                self.activations.append(act_map[activation]())

    def forward(self, x):
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x)
        return x


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Linear(1, 10))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(10, 1))
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    model = net()
    print('number of parameters: {}'.format(sum(p.numel() for p in model.parameters())))