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


# Optimizer

lr_init = 1e-2
weight_decay = 0.02
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay= weight_decay)

max_iters = 1000

eval_interval = 50

warmup_iters = 10
warmup_factor = 1e-3

lr_final = 1e-5

def lr_lambda(current_iter):
    if current_iter < warmup_iters:
        return warmup_factor + (1 - warmup_factor) * current_iter / warmup_iters
    else:
        decay_iters = max_iters - warmup_iters
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - warmup_iters) / decay_iters))
        return max(cosine_decay, lr_final / lr_init)

scheduler = torch.optim.lr_scheduler.lambdaLR(optimizer, lr_lambda)

# training script


