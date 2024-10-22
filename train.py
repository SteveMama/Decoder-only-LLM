import os
from main import *
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import math
from tiny_shakespeare_tokenizer import *
tokenizer = get_tokenizer(size = 512)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(text[:200])

chars = sorted(list(set(text)))
v= len(chars)
print(chars)
print(v)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# Training the model

params = ModelArgs()
print(params)
model = Llama3(params, tokenizer).to(params.device)

print(model)


# preparing a batch

def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - params.max_seq_len, (batch_size,))
    x = torch.stack([data[i:i + params.max_seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + params.max_seq_len + 1] for i in ix])
    x, y = x.to(params.device), y.to(params.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters = 5):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, targets = Y)
            loss[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



