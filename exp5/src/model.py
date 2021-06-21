import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, PairNorm
from torch_geometric.utils import dropout_adj


activations = {
    'relu': torch.relu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
}


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 n_layers: int, act: str = 'relu', add_self_loops: bool = True,
                 pair_norm: bool = True, dropout: float = .0, drop_edge: float = .0):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.drop_edge = drop_edge
        self.pair_norm = pair_norm
        self.act = activations[act] if isinstance(act, str) else act

        self.conv_list = torch.nn.ModuleList()
        for i in range(n_layers):
            in_c, out_c = hidden_channels, hidden_channels
            if i == 0:
                in_c = in_channels
            elif i == n_layers - 1:
                out_c = num_classes
            self.conv_list.append(GCNConv(in_c, out_c, add_self_loops=add_self_loops))

    def forward(self, x, edge_index):
        edge_index, _ = dropout_adj(edge_index, p=self.drop_edge)

        for i, conv in enumerate(self.conv_list):
            x = conv(x, edge_index)
            if self.pair_norm:
                x = PairNorm()(x)
            if i < len(self.conv_list) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
