raw_data_folder = './aclImdb/'

vectors_folder = './glove.6B/'

data_folder = './data/'

SEED = 2077
DEVICE = 'cuda:0'

VAL_RATIO = 0.2

VECTORS = 'glove.6B.100d'

VOCAB_SIZE = 400000
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
N_LAYERS = 1
BIDIRECTIONAL = False
DROPOUT = 0.
BATCH_SIZE = 128
N_EPOCHS = 10
MODEL_BASE = 'RNN'
