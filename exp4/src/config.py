raw_data_folder = './aclImdb/'
vectors_folder = './glove.6B/'
data_folder = './data/'


SEED = 2077
DEVICE = 'cuda:0'

VAL_RATIO = 0.2

HIDDEN_DIM = 32
BATCH_SIZE = 8
N_EPOCHS = 10
MAX_LEN = 256
LEARNING_RATE = 1e-5
FREEZE_BERT = True
JUST_TEST = False
