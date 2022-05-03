from torch import nn
import os
import torch
from torch.nn import LayerNorm
import math
from torch.nn import init, GRUCell


class LayerNormGRUCell(torch.nn.Module):
    '''
    This part comes from https://github.com/ElektrischesSchaf/LayerNorm_GRU/blob/main/GRU_layernorm_cell.py
    '''
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        self.ln_cell_1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):  # bs,hid
        h = h
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1(h_hat_first_half)
        h_hat_last_half = self.ln_cell_2(h_hat_last_half)

        h_hat = torch.tanh(h_hat_first_half + torch.mul(r_t, h_hat_last_half))
        h_t = torch.mul(1 - z_t, h) + torch.mul(z_t, h_hat)
        h_t = h_t.view(h_t.size(0), -1)
        return h_t


class LayerNormGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, gru_ln):
        super(LayerNormGRU, self).__init__()
        if gru_ln:
            self.gru_cell = LayerNormGRUCell(input_size, hidden_size, bias=True)
        else:
            self.gru_cell = GRUCell(input_size, hidden_size, bias=True)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, input, length):
        '''
        :param input: bs,len,hidden
        :param length: bs
        :return:
        '''
        bs, l = input.shape[0], input.shape[1]
        h_x = torch.zeros((bs, self.hidden_size)).to(input.device)
        output = []
        for i in range(l):
            h_x = self.gru_cell(input[:, i, :], h_x)
            output.append(h_x)
        output = torch.stack(output, dim=1)  # bs,len,hidden
        ind = length - 1
        return torch.gather(output, 1, ind.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, output.shape[-1])).unsqueeze(1)
        # bs,hidden


class PathEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.ap_split:
            self.ap_embedding = nn.Embedding(self.args.path_embedding_num + 1, self.args.path_embedding_size,
                                             padding_idx=self.args.path_embedding_num)
            init.xavier_normal_(self.ap_embedding.weight)
        self.embedding = nn.Embedding(self.args.path_embedding_num + 1, self.args.path_embedding_size,
                                      padding_idx=self.args.path_embedding_num)
        init.xavier_normal_(self.embedding.weight)
        self.gru_size = self.args.gru_size
        self.layers = self.args.gru_layers
        self.num_directions = 2 if self.args.bidirectional else 1

        if self.args.relation_path:
            self.rp_rnn = LayerNormGRU(self.args.path_embedding_size, self.gru_size, self.args.gru_ln)
        else:
            self.rp_rnn = None
        if self.args.absolute_path:
            self.ap_rnn = LayerNormGRU(self.args.path_embedding_size, self.gru_size, self.args.gru_ln)
        else:
            self.ap_rnn = None
        if self.args.gru_ln:
            self.gru_ln = LayerNorm(2 * self.gru_size)

    def forward(self, paths, paths_mask, type='relation'):
        '''
        :param type:
        :param paths: bs,max_path_num,max_path_length
        :param paths_mask: bs,max_path_num
        :return:bs,max_path_num,hidden
        '''
        assert type in ['relation', 'absolute']
        if type == 'relation':
            p_1 = self.embedding(paths)
            # bs,max_path_num,max_path_length,dim
        elif type == 'absolute':
            if self.args.ap_split:
                p_1 = self.ap_embedding(paths)
                # bs,max_path_num,max_path_length,dim
            else:
                p_1 = self.embedding(paths)
                # bs,max_path_num,max_path_length,dim
        else:
            raise Exception('Not Valid Path Type !')
        bs, max_path_num, max_path_length, dim = p_1.shape

        input = p_1.view(-1, max_path_length, dim)
        # bs*max_path_num,max_path_length,dim

        length = paths_mask.view(-1)
        # bs*max_path_num

        if type == 'relation':
            output = self.rp_rnn(input, length).view(bs, max_path_num, -1)  # output: bs,max_path_num,hidden
            forward_ = torch.cat((output[:, 0::2, :], output[:, 1::2, :]), dim=-1)  # bs,256,128
            backward_ = torch.cat((output[:, 1::2, :], output[:, 0::2, :]), dim=-1)
            output = torch.zeros((bs, max_path_num, self.gru_size * 2)).to(output.device)  # bs,512,128
            output[:, 0::2, :] = forward_
            output[:, 1::2, :] = backward_
            # bs,max_path_num,gru_size*2
        elif type == 'absolute':
            output = self.ap_rnn(input, length).view(bs, max_path_num, -1)  # output: bs,max_path_num,hidden
            output = torch.cat((output[:, 0::2, :], output[:, 1::2, :]), dim=-1)
            # bs,max_path_num/2,gru_size*2
        else:
            raise Exception('Not Valid Path Type !')
        # return self.gru_norm(output)
        if self.args.gru_ln:
            output = self.gru_ln(output)
        return output

