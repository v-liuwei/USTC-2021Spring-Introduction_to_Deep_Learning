import os
import torch
from tqdm import tqdm
from config import *
import re
import pickle
from transformers import BertTokenizer
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def get_raw_data(which):
    cache_file = os.path.join(data_folder, which + '_raw.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data

    texts = []
    labels = []
    raw_folder = os.path.join(raw_data_folder, which)
    for subfolder in ['pos', 'neg']:
        folder_name = os.path.join(raw_folder, subfolder)
        for file in tqdm(os.listdir(folder_name),
                         bar_format='{percentage:3.0f}%|{bar:20}{r_bar}'):
            with open(os.path.join(folder_name, file), 'rb') as f:
                text = f.read().decode('utf-8')
                label = 1 if subfolder == 'pos' else 0
                texts.append(text)
                labels.append(label)
    data = (texts, labels)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    return data


def text_preprocessing(text):
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def preprocessing_for_bert(sentences):
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    for sent in tqdm(sentences):
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            truncation=True,
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def get_dataloader():
    dataset_cache_path = data_folder + 'train_data.pkl'
    dataset = dict()
    if not os.path.exists(dataset_cache_path):
        train_texts, train_labels = get_raw_data('train')
        test_texts, test_labels = get_raw_data('test')

        indices = np.random.permutation(len(train_texts))
        val_indices, train_indices = np.split(indices, [round(len(train_texts) * VAL_RATIO)])
        train_texts, train_labels = np.array(train_texts), np.array(train_labels)
        val_texts, val_labels = train_texts[val_indices], train_labels[val_indices]
        train_texts, train_labels = train_texts[train_indices], train_labels[train_indices]

        train_inputs, train_masks = preprocessing_for_bert(train_texts)
        val_inputs, val_masks = preprocessing_for_bert(val_texts)
        test_inputs, test_masks = preprocessing_for_bert(test_texts)

        # Convert other data types to torch.Tensor
        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)
        test_labels = torch.tensor(test_labels)

        dataset['train'] = TensorDataset(train_inputs, train_masks, train_labels)
        dataset['val'] = TensorDataset(val_inputs, val_masks, val_labels)
        dataset['test'] = TensorDataset(test_inputs, test_masks, test_labels)
        for which in ['train', 'val', 'test']:
            pickle.dump(dataset[which], open(os.path.join(data_folder, f'{which}_data.pkl'), 'wb'))
    else:
        for which in ['train', 'val', 'test']:
            dataset[which] = pickle.load(open(os.path.join(data_folder, f'{which}_data.pkl'), 'rb'))

    train_sampler = RandomSampler(dataset['train'])
    train_dataloader = DataLoader(dataset['train'], sampler=train_sampler, batch_size=BATCH_SIZE)

    val_sampler = SequentialSampler(dataset['val'])
    val_dataloader = DataLoader(dataset['val'], sampler=val_sampler, batch_size=BATCH_SIZE)

    test_sampler = SequentialSampler(dataset['test'])
    test_dataloader = DataLoader(dataset['test'], sampler=test_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    get_dataloader()
