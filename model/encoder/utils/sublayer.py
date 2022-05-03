import torch.nn as nn
from torch.nn import LayerNorm


class SublayerConnection(nn.Module):

    def __init__(self, args):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(args.hidden)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
