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

