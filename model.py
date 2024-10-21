import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from typing import Optional, Tuple

import time

import os
import json

from tiny_shakespeare_tokenizer import *
tokenizer = get_tokenizer(size=512)

#setting up the parameters

@dataclass
class ModelArgs:
    dim: int = 128
    n_layers: int = 8
    n_heads: int = 4
    n_kv_heads: int = 4
    vocab_size: int = tokenizer.vocab_len
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_batch_size: int = 32
    max_seq_len: int = 512
    device: str ='cuda'
    dropout_rate: float = 0.1


# RMS Norm

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps:float = 1e-6):
        super().__init_()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.sqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# RoPE

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device = freqs.device, dtype = torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to(params.device)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'freqs_cis.shape{freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1])}
    shape = [d if i == 1 or i == ndim -1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk : torch.Tensor,
        freqs_cis : torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1 , 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# Attention module

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """ torch.repeat_interleave(x, dim = 2, repeats=n_rep)"""
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, seqlen, n_kv_heads, n_rep, head_dim).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim,args.dim, bias = False)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            requires_grad=False
        ).to(args.device)

        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self. n_kv_heads, self.head_dim),
            requires_grad= False
        ).to(args.device)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            start_pos: int = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if start_pos is not None:

            self.cache_k = self.cache_k.to(xq)




