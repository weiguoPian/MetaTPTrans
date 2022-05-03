import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class RelationAwareAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.path_value = self.args.path_value

    def forward(self, query, key, value, r_k=None, r_v=None, path_map=None, mask=None, dropout=None, ap=None):
        """
        :param r_v: bs,h,max_path_num,hidden//head
        :param r_k: bs,h,max_path_num,hidden//head
        :param ap: bs, h, max_code_length, max_code_length
        :param path_map: bs,max_code_length,max_code_length
        :param query: bs, head,max_code_length, hidden//head
        :param key:
        :param value:bs, h,max_code_length, hidden
        :param mask:bs, 1,max_code_length,max_code_length
        :param dropout:
        :return:
        """

        score = torch.einsum('bhik,bhjk->bhij', query, key)

        bs, h, max_code_length, dim = query.shape

        if r_k is not None:
            r_k = torch.cat((r_k, torch.zeros(1, 1, 1, 1).expand(bs, h, -1, dim).to(r_k.device)), dim=2)
            # relation: bs,h,max_path_num+1,dim => bs,h,(max_path_num+1),dim
            score_r = torch.matmul(query, r_k.transpose(-1, -2)).gather(-1, path_map.unsqueeze(1))
            score += score_r

        if ap is not None:
            score += ap
        scores = score / math.sqrt(query.size(-1) * self.args.sqrt_norm)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        attn_sum = torch.einsum('bhij,bhjk->bhik', p_attn, value)

        if r_v is not None and self.path_value:
            # print(r_v.shape)
            r_v = torch.cat((r_v, torch.zeros(1, 1, 1, 1).expand(bs, h, -1, dim).to(r_v.device)), dim=2)
            bs, h, max_path_num, dim = r_v.shape
            r_attn_sum = torch.zeros(bs, h, max_code_length, max_path_num).to(r_v.device). \
                scatter_add_(-1, path_map.unsqueeze(1).expand(-1, h, -1, -1), p_attn).matmul(r_v)
            attn_sum += r_attn_sum
        return attn_sum, p_attn
