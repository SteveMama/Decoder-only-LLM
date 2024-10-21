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
toe