import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, dim, length):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(length, dim).float()
        pe.require_grad = False

        position = torch.arange(0, length).float().unsqueeze(1)
        div_term = (torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        bs, len, dim = x.shape
        return self.pe[:, :len, :dim]
