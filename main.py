import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import math


v = 10

seq_len = 5

b = 1

tokens = torch.randint(v, (b, seq_len))
print(tokens.shape, tokens)

# Intialise residual State

d = 16
embedding = nn.Embedding(v, d)
print(embedding.weight.shape,embedding.weight)