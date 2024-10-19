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

x = embedding(tokens)
print(x.shape, x)


# Precompting RoPE Frequences

theta = 10000
num_heads = 4
head_dim = d // num_heads

freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim //2)].float() / head_dim))
print(f'freqs: {freqs.shape}\n {freqs}\n')

t = torch.arange(seq_len * 2, device = freqs.device, dtype = torch.float32)
print(f'freqs: {freqs.shape}\n{t}\n')

freqs = torch.outer(t, freqs)
print(f'freqs: {freqs.shape}\n{freqs}\n')

freqs_cis = torch.polar(torch.ones_like(freqs), freqs)[: seq_len]
print(f'freqs_cis:{freqs_cis.shape}\n{freqs_cis}')


# Percomputing Causal Mask

mask = torch.full(
    (seq_len, seq_len),
    float("-inf")
)

mask = torch.triu(mask, diagonal=1)
print(mask)

# Normalisation

h = x
print(f'h: {h.shape}\n{h}')

mean_squared = x.pow(2).mean(dim=-1, keepdim = True)
print(mean_squared)