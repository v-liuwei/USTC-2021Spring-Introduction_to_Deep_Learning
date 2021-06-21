import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 n_layers: int = 1, bidirectional: bool = False,
                 dropout: float = 0., model_base: str = 'RNN'):
        super(RNNClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.model_base = model_base.lower()
        if self.model_base == 'lstm':
            model = nn.LSTM
        else:
            model = nn.RNN

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = model(embedding_dim,
                         hidden_dim,
                         num_layers=n_layers,
                         bidirectional=bidirectional,
                         dropout=dropout)
        if self.bidirectional:
            hidden_dim *= 2
        self.fc = nn.Linear(hidden_dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, x, x_len):
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_len)
        if self.model_base == 'lstm':
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)  # h_n.shape = (num_layers * num_directions, batch, hidden_size)
        if self.bidirectional:
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # get last layer
        else:
            hidden = h_n[-1]
        logits = self.fc(hidden)
        output = self.act(logits)
        return output
