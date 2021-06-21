from utils import binary_acc, set_seed
from config import *
import spacy
import torch
import torch.nn as nn
from torch.optim import Adam
from data import TEXT, get_dataloader
import random
import os
from model import RNNClassifier
from utils import binary_acc
import numpy as np
from tqdm import tqdm


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    pbar = tqdm(iterator, unit='batch', ascii=True,
                bar_format='{percentage:3.0f}%|{bar:20}{r_bar}')
    model.train()
    for batch in pbar:
        optimizer.zero_grad()

        (text, text_lengths), label = batch.text, batch.label
        text = text.to(DEVICE)
        label = label.to(DEVICE)

        preds = model(text, text_lengths).squeeze()
        loss = criterion(preds, label.float())
        loss.backward()
        optimizer.step()

        acc = binary_acc(preds, label)

        pbar.set_postfix({'train_loss': loss.item(),
                          'train_acc': acc.item()})

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            (text, text_lengths), label = batch.text, batch.label
            text = text.to(DEVICE)
            label = label.to(DEVICE)

            preds = model(text, text_lengths).squeeze()
            loss = criterion(preds, label.float())
            acc = binary_acc(preds, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == '__main__':
    set_seed(SEED)
    train_iterator, val_iterator, test_iterator = get_dataloader()
    rnn = RNNClassifier(
        vocab_size=len(TEXT.vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        model_base=MODEL_BASE
    )
    pretrained_embeddings = TEXT.vocab.vectors
    rnn.embedding.weight.data.copy_(pretrained_embeddings)
    optimizer = Adam(rnn.parameters())
    criterion = nn.BCELoss()

    rnn = rnn.to(DEVICE)
    criterion = criterion.to(DEVICE)

    best_valid_loss = np.inf
    for epoch in range(N_EPOCHS):
        print(f">>> Epoch {epoch+1}/{N_EPOCHS}")

        train_loss, train_acc = train(rnn, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(rnn, val_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(rnn.state_dict(), './checkpoint/best_weights.pt')

        print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc * 100:.2f}%')
    print(">>> Testing...")
    rnn.load_state_dict(torch.load("./checkpoint/best_weights.pt"))
    test_loss, test_acc = evaluate(rnn, test_iterator, criterion)
    print(f'\tTest Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%')
