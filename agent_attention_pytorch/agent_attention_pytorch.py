import torch
from torch.nn import Module
from torch import nn, einsum, Tensor

from einops import rearrange

# functions

def exists(v):
    return v is not None

# main class

class AgentAttention(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
