default_cfg = {
    'data_root': './../GNN/',
    'data_name': 'cora',
    'num_train_per_class': 20,
    'num_val': 500,
    'num_test': 1000,
    'seed': 114514,
    'device': 'cuda:0',
    'epochs': 1000,
    'patience': 5,
    'lr': 5e-3,
    'weight_decay': 5e-4,
    'hidden_dim': 32,
    'n_layers': 2,
    'activations': 'relu',
    'dropout': 0.5,
    'drop_edge': 0.,
    'add_self_loop': True,
    'pair_norm': False,
    'test_only': False
}


class Config(object):
    def __init__(self, ):
        self.data_root = None
        self.data_name = None
        self.num_train_per_class = None
        self.num_val = None
        self.num_test = None
        self.seed = None
        self.device = None
        self.epochs = None
        self.patience = None
        self.lr = None
        self.weight_decay = None
        self.hidden_dim = None
        self.n_layers = None
        self.activations = None
        self.dropout = None
        self.drop_edge = None
        self.add_self_loop = None
        self.pair_norm = None
        self.test_only = None
        self.reset()

    def reset(self):
        for key, val in default_cfg.items():
            setattr(self, key, val)

    def update(self, new_cfg):
        for key, val in new_cfg.items():
            setattr(self, key, val)
