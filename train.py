import os
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

