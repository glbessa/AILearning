import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()

        