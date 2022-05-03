import torch.nn as nn
from .single import RelationAwareAttention


class MultiHeadedAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.hidden
        self.h = args.attn_heads
        assert self.d_model % self.h == 0
        self.d_k = args.hidden // args.attn_heads
        # We assume d_v always equals d_k
        self.linear_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(3)])
        self.output_linear = nn.Linear(self.d_model, self.d_model)
        self.attention = RelationAwareAttention(args)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, query, key, value, mask=None, r_k=None, r_v=None, path_map=None, ap=None):
        '''
        :param ap: bs, h, max_code_length, max_code_length
        :param path_map: bs,max_code_length,max_code_length
        :param query: bs, max_code_length, hidden
        :param key: bs, max_code_length, hidden
        :param value: bs, max_code_length, hidden
        :param r_k: bs, h, max_path_num,hidden//heads
        :param r_v: bs, h, max_path_num,hidden//heads
        :param mask:bs, 1,max_code_length,max_code_length
        :return:
        '''
        batch_size, max_code_length = query.size(0), query.size(1)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # bs,l,dim -> bs,l,head,d_k -> bs,head,l,d_k

        x, attn = self.attention(query, key, value, path_map=path_map, mask=mask,
                                 dropout=self.dropout, ap=ap, r_k=r_k, r_v=r_v)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
