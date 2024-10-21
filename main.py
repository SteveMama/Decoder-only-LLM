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
xq = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xk = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
print(f'xq: {xq.shape}\n{xq}\n')
print(f'xq: {xk.shape}\n{xq}\n')

ndim = xq.ndim
assert 0 <= 1 < ndim
assert freqs_cis.shape == (xq.shape[1], xq.shape[-1]), f'freqs_cis.shape {freqs_cis.shape} != xq.shape[1], xq.shape[-1] {(xq.shape[1], xq.shape[-1])}'

shape = [d if i == 1 or i == xq.ndim - 1 else 1 for i, d in enumerate(xq.shape)]
print(f'shape: {shape}\n')

freqs_cis = freqs_cis.view(*shape)
print(f'freqs_cis: {freqs_cis.shape}\n{freqs_cis}')

# multiply data * freq
xq = torch.view_as_real(xq * freqs_cis).flatten(3).type_as(xv)
xk = torch.view_as_real(xk * freqs_cis).flatten(3).type_as(xv)
print(f'xq: {xq.shape}\n{xq}\n')
print(f'xk: {xk.shape}\n {xk}\n')


# Calculating Self-Attention

if num_kv_heads != num_heads:
    num_queries_per_kv = num_heads // num_kv_heads
    xk = torch.repeat_interleave(xk, num_queries_per_kv, dim = 2)
    xv = torch.repeat_interleave(xv, num_queries_per_kv, dim =2)

print(xq.shape, xk.shape, xv.shape)

xq = xq.transpose(1, 2)
xk = xk.transpose(1, 2)
xv = xv.transpose(1, 2)

scores = torch.matmul(xq, xk.transpose(2,3))

scores = scores / math.sqrt(head_dim)

print(scores.shape, scores)

# Masks

scores = scores + mask
print(scores.shape, scores)

scores = F.softmax(scores.float(), dim =-1).type_as(xq)
print(scores)

output = torch.matmul(scores, xv)
print(output.shape, output)

output = output.transpose(1,2).contiguous().view(b, seq_len, -1)
print(output.shape, output)

wo = nn.Linear(num_heads * head_dim, d, bias = False)
Xout = wo(output)
print(Xout.shape, Xout)

# Building Residual Connection

post_attn_norm = RMSNorm(d)
h += post_attn_norm(Xout)
print(h.shape, h)

pre_ffwd_norm = RMSNorm(d)
h_normed = pre_ffwd_norm(h)

# Implementing SwiGLU Feedforward Network

hidden_dim = 4 * d
print(hidden_dim)
hidden_dim = int(2 * hidden_dim / 3)
print(hidden_dim)
multiple_of = 256
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
print(hidden_dim)

up = nn.Linear(d, hidden_dim, bias = False)
gate = nn.Linear(d, hidden_dim, bias = False)
down = nn.Linear(hidden_dim, d, bias = False)

up_proj = up(h_normed)
print(up_proj.shape, up_proj)

gate_proj = F.silu(gate(h_normed))
print(gate_proj.shape, gate_proj)

ffwd_output = down(up_proj * gate_proj)
print(ffwd_output.shape, ffwd_output)

out = h + ffwd_output
print(out.shape, out)

# Output calculation

final_norm = RMSNorm(d)
out_normed = final_norm(out)

final_output = nn.Linear(d, v, bias = False)
logits = final_output(out_normed).float()
print(logits.shape, logits)

probs = F.softmax(logits, dim =-1)
print(probs)

greedy_indices = torch.argmax(probs, dim = -1)
print(greedy_indices)

# Loss functions

target_token_indices = torch.randint(0, v, greedy_indices.shape)
print(target_token_indices)

loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(logits.view(1, v, seq_len), target_token_indices)
print(loss)


# SwiGLU Activation function

def swiGLU(x):
    deno = (1 + math.exp(-1.702 * x))
    y = (1 / deno)

    return y

print(swiGLU(0.52))