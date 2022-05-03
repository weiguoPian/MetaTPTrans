import torch
import torch.nn as nn
from .gelu import GELU
from torch.nn import ReLU

class metaLinearLayer(nn.Module):
    def __init__(self, args, input_dim, out_dim, bias=True):
        super().__init__()
        self.args = args
        self.bias = bias
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.left = nn.Parameter(torch.empty((self.args.projection_dim, self.input_dim)))
        self.right = nn.Parameter(torch.empty(self.args.projection_dim, self.out_dim))
        nn.init.kaiming_uniform_(self.left)
        nn.init.kaiming_uniform_(self.right)
        
        if self.bias:
            self.project2bias = nn.Linear(self.args.projection_dim, self.out_dim)
        
        self.activation = GELU() if args.activation == 'gelu' else ReLU()

        self.eye = torch.eye(self.args.projection_dim).unsqueeze(0)
        self.eye = nn.Parameter(self.eye, requires_grad=False)


    # (BS, in)
    def forward(self, input, meta_input):
        # (BS, L, L)
        diag_language_embed = torch.unsqueeze(meta_input, -1) * self.eye

        meta_weight = torch.matmul(torch.matmul(diag_language_embed, self.left).transpose(1, 2), self.right)

        if len(input.shape) == 2:
            out = torch.einsum("ij,ijk->ik", [input, meta_weight])
            if self.bias:
                meta_bias = self.project2bias(meta_input)
                out += meta_bias

        elif len(input.shape) == 3:
            out = torch.einsum("imj,ijk->imk", [input, meta_weight])
            if self.bias:
                meta_bias = self.project2bias(meta_input)
                meta_bias = meta_bias.unsqueeze(1).repeat(1, out.shape[1], 1)
                out += meta_bias

        elif len(input.shape) == 4:
            out = torch.einsum("imnj,ijk->imnk", [input, meta_weight])
            if self.bias:
                meta_bias = self.project2bias(meta_input)
                meta_bias = meta_bias.unsqueeze(1).unsqueeze(1).repeat(1, out.shape[1], out.shape[2], 1)
                out += meta_bias
        else:
            print(input.shape)
            print(meta_weight.shape)
        
        return out