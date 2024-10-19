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

x_normed = x * torch.rsqrt(mean_squared + 1e-6)
print(f'x_normed: {x_normed.shape}\n{x_normed }')

# RMS Scale

rms_scale = torch.ones(d)
print(f'rms_scale:{rms_scale.shape}\n{rms_scale}\n')

x_normed *= rms_scale
print(f'x_normed: {x_normed.shape}\n{x_normed}')

# RMSNorm

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

## Multi-Query Attention

print(h, x_normed)

num_kv_heads = 2
assert num_heads % num_kv_heads ==0
print(f"as a reminder: num_heads = {num_heads}, head_dim = {head_dim}")


# initialisation of Self-attention weight matrices

wq = nn.Linear(d, num_heads * head_dim, bias = False)
wk = nn.Linear(d, num_kv_heads * head_dim, bias = False)
wv = nn.Linear(d, num_kv_heads * head_dim, bias = False)
print("Attention Weights: ", wq.weight.shape, wk.weight.shape, wv.weight.shape)

xq = wq(x_normed)
xk = wk(x_normed)
xv = wv(x_normed)
print("Attention Projections: ", xq.shape, xk.shape, xv.shape)

xq = xq.view(b, seq_len, num_heads, head_dim)
xk = xk.view(b, seq_len, num_kv_heads, head_dim)
xv = xv.view(b, seq_len, num_kv_heads, head_dim)
print("Reshaped: ", xq.shape, xk.shape, xv.shape)

# Rotary Position Embeddings