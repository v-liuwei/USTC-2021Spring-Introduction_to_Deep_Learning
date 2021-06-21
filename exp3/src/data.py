from torchtext.legacy import data
import os
import torch
from tqdm import tqdm
from config import *
import spacy
import pickle


_nlp = spacy.load('en_core_web_sm')
TEXT = data.Field(tokenize=lambda x: [t.text for t in _nlp(x)],
                  include_lengths=True, lower=True)
LABEL = data.LabelField(use_vocab=False)


def _get_examples(which, fields=None):
    cache_file = os.path.join(data_folder, which + '.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            examples = pickle.load(f)
        return examples

    examples = []
    raw_folder = os.path.join(raw_data_folder, which)
    for subfolder in ['pos', 'neg']:
        folder_name = os.path.join(raw_folder, subfolder)
        for file in tqdm(os.listdir(folder_name),
                         bar_format='{percentage:3.0f}%|{bar:20}{r_bar}'):
            with open(os.path.join(folder_name, file), 'rb') as f:
                text = f.read().decode('utf-8').replace('\n', '').lower()
                label = 1 if subfolder == 'pos' else 0
                examples.append(data.Example.fromlist([text, label], fields))
    with open(cache_file, 'wb') as f:
        pickle.dump(examples, f)
    return examples


def get_dataloader(text_field=TEXT, label_field=LABEL):

    fields = [('text', text_field), ('label', label_field)]

    train_data = _get_examples('train', fields)
    test_data = _get_examples('test', fields)
    train_data = data.Dataset(train_data, fields)
    test_data = data.Dataset(test_data, fields)
    val_data = None
    if VAL_RATIO:
        train_data, val_data = train_data.split(split_ratio=1 - VAL_RATIO)
    vectors = VECTORS.replace('.txt', '')
    text_field.build_vocab(train_data, max_size=VOCAB_SIZE,
                           vectors=vectors, vectors_cache=vectors_folder)
    label_field.build_vocab(train_data)
    if VAL_RATIO:
        train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size=BATCH_SIZE,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
        )
        return train_iterator, val_iterator, test_iterator
    else:
        train_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=BATCH_SIZE,
            sort=False
        )
        return train_iterator, test_iterator
