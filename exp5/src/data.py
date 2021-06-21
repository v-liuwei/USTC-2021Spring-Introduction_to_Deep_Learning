import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected
import os


classes = {
    'citeseer': ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'],
    'cora': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods',
             'Reinforcement_Learning', 'Rule_Learning', 'Theory']
}


class NodeClsDataset(InMemoryDataset):
    def __init__(self, root, name, num_train_per_class: int = 20,
                 num_val: int = 500, num_test: int = 1000, transform=None):
        self.name = name.lower()
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        super(NodeClsDataset, self).__init__(root, transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.name}.content', f'{self.name}.cites']

    @property
    def processed_file_names(self):
        return [f'{self.name}.pt']

    def download(self):
        pass

    def process(self):
        label2index = {label: i for i, label in enumerate(classes[f'{self.name}'])}
        id2index, x, y = read_content(self.raw_paths[0], label2index)
        edge_index = read_cites(self.raw_paths[1], id2index)
        data = Data(x=x, y=y, edge_index=edge_index)

        data.train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        data.val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        data.test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        for c in range(len(label2index)):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))[:self.num_train_per_class]]
            data.train_mask[idx] = True

        remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        data.val_mask[remaining[:self.num_val]] = True
        data.test_mask[remaining[self.num_val:self.num_val + self.num_test]] = True

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])


def read_content(content_file, label2index):
    with open(content_file, 'r') as f:
        lines = f.read().strip().split('\n')[:-1]
    id2index = {}
    x = []
    y = []
    for i, line in enumerate(lines):
        line = line.strip().split('\t')
        paper_id, attr, label = line[0], line[1:-1], line[-1]
        id2index[paper_id] = i
        x.append([float(e) for e in attr])
        y.append(label2index[label])
    return id2index, torch.tensor(x), torch.tensor(y, dtype=torch.long)


def read_cites(cites_file, id2index):
    with open(cites_file, 'r') as f:
        lines = f.read().strip().split('\n')[:-1]
    edge_index = []
    for line in lines:
        cited, citing = line.strip().split('\t')
        if citing not in id2index or cited not in id2index:
            continue
        id_cited, id_citing = id2index[cited], id2index[citing]
        edge_index.append([id_citing, id_cited])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = to_undirected(edge_index)
    return edge_index.t().contiguous()
