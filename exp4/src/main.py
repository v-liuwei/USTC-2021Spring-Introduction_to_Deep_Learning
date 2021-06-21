from utils import set_seed
from config import *
import torch
import torch.nn as nn
from torch.optim import Adam
from data import get_dataloader
import random
import os
import numpy as np
from tqdm import tqdm
from model import BertClassifier


def train(model, dataloader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    pbar = tqdm(dataloader, unit='batch', ascii=True,
                bar_format='{percentage:3.0f}%|{bar:20}{r_bar}')
    model.train()
    for batch in pbar:
        optimizer.zero_grad()

        input_ids, attn_mask, labels = tuple(t.to(DEVICE) for t in batch)

        logits = model(input_ids, attn_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        acc = (preds == labels).cpu().numpy().mean()

        pbar.set_postfix({'train_loss': loss.item(),
                          'train_acc': acc.item()})

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def evaluate(model, dataloader, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attn_mask, labels = tuple(t.to(DEVICE) for t in batch)

            logits = model(input_ids, attn_mask)
            loss = criterion(logits, labels)

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()
            acc = (preds == labels).cpu().numpy().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


if __name__ == '__main__':
    set_seed(SEED)
    train_dataloader, val_dataloader, test_dataloader = get_dataloader()
    bert = BertClassifier(freeze_bert=FREEZE_BERT)
    # pretrained_embeddings = TEXT.vocab.vectors
    # rnn.embedding.weight.data.copy_(pretrained_embeddings)
    optimizer = Adam(bert.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    bert = bert.to(DEVICE)
    criterion = criterion.to(DEVICE)
    if not JUST_TEST:
        best_valid_loss = np.inf
        for epoch in range(N_EPOCHS):
            print(f">>> Epoch {epoch+1}/{N_EPOCHS}")

            train_loss, train_acc = train(bert, train_dataloader, optimizer, criterion)
            valid_loss, valid_acc = evaluate(bert, val_dataloader, criterion)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(bert.state_dict(), './checkpoint/best_weights.pt')

            print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc * 100:.2f}%')
    print(">>> Testing...")
    bert.load_state_dict(torch.load("./checkpoint/best_weights.pt"))
    test_loss, test_acc = evaluate(bert, test_dataloader, criterion)
    print(f'\tTest Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%')
