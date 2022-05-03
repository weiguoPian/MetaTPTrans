from torch import nn
import os
import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm
from .positional import PositionalEmbedding
import math
from torch.nn import init


class TokenEmbedding(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.args = args
        if self.args.pretrain:
            self.get_embedding()
            self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, padding_idx=self.vocab.pad_index,
                                                          freeze=False)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.args.embedding_size, padding_idx=self.vocab.pad_index)
            init.xavier_normal_(self.embedding.weight)
        if args.embedding_size != args.hidden:
            self.in_ = nn.Linear(args.embedding_size, args.hidden)
        else:
            self.in_ = None

    def get_embedding(self):
        embedding_dir = './catch/{}'.format(self.args.dataset)
        if not os.path.exists(embedding_dir):
            os.makedirs(os.path.join(embedding_dir))
        embedding_path = './catch/{}/{}_embedding.pkl'.format(self.args.dataset, self.vocab.type)
        if os.path.exists(embedding_path):
            with open(embedding_path, 'rb') as f:
                embedding = pkl.load(f)
                print('Load Embedding from Catch')
                self.embedding_matrix = embedding.clone().detach()
                assert len(embedding) == self.vocab_size
        else:
            with open(self.args.embedding_file, 'r') as f:
                line = f.readline().strip().split()
                embedding_dim = len(line) - 1
            self.embedding_matrix = torch.randn(self.vocab_size, self.args.embedding_size)
            count = 0
            with open(self.args.embedding_file, 'r', encoding='utf-8') as f:
                print('Create {} Embedding from raw data'.format(self.vocab.type))
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split()
                    word = line[0]
                    idx = self.vocab.find(word)
                    if idx != self.vocab.unk_index:
                        vector = torch.tensor(np.append(np.array(line[1:]),
                                                        np.array(
                                                            [0] * (self.args.embedding_size - embedding_dim))).astype(
                            np.float))
                        self.embedding_matrix[idx] = vector
                        count += 1
                print('Pretrain Word = {} for {}'.format((count / self.vocab_size), self.vocab.type))
            with open(embedding_path, 'wb') as f:
                pkl.dump(self.embedding_matrix, f)
            print('Save {} Embedding into catch'.format(self.vocab.type))


class LeftEmbedding(TokenEmbedding):
    '''
    The embedding for encoder, so is the ''left''
    '''
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        if args.absolute_position:
            self.p = PositionalEmbedding(args.embedding_size, args.max_code_length)
        else:
            self.p = None
        if args.is_named:
            self.n = nn.Embedding(3, args.embedding_size, padding_idx=2)
            init.xavier_normal_(self.n.weight)
        else:
            self.n = None
        self.args = args

    def forward(self, content, named=None):
        '''
        :param named:
        :param content: bs,max_code_length
        '''
        c_1 = self.embedding(content)
        if self.args.embedding_mul:
            c_1 *= math.sqrt(self.args.embedding_size)
        if self.p:
            c_1 = c_1 + self.p(c_1)
        if self.n:
            c_1 = c_1 + self.n(named)
        if self.in_:
            c_1 = self.in_(c_1)
        return c_1


class RightEmbedding(TokenEmbedding):
    '''
    The embedding for decoder, so is the ''right''
    '''
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.out = nn.Linear(args.embedding_size, self.vocab_size)
        if args.embedding_size != args.hidden:
            self.out_ = nn.Linear(args.hidden, args.embedding_size)
        else:
            self.out_ = None
        if args.absolute_position:
            self.p = PositionalEmbedding(args.embedding_size, args.max_code_length)
        else:
            self.p = None
        self.args = args

    def forward(self, f_source):
        '''
        :param f_source: bs,max_target_len
        :return:bs,max_target_len,hidden
        '''
        c_1 = self.embedding(f_source)
        if self.args.embedding_mul:
            c_1 *= math.sqrt(self.args.embedding_size)
        if self.p:
            c_1 = c_1 + self.p(c_1)
        if self.in_:
            c_1 = self.in_(c_1)
        return c_1

    def prob(self, data):
        if self.out_:
            data = self.out_(data)
        return self.out(data)
