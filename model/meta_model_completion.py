from torch import embedding, nn
from .embedding import LeftEmbedding, RightEmbedding, PathEmbedding
from .encoder import MetaEncoder
import torch
import math
import torch.nn.functional as F
from .encoder.utils.gelu import GELU

class metaModelCompletion(nn.Module):
    def __init__(self, args, s_vocab, num_classes):
        super().__init__()
        self.args = args
        self.left_embedding = LeftEmbedding(args, s_vocab)
        if args.relation_path or args.absolute_path:
            self.path_embedding = PathEmbedding(args)

        self.encoder = MetaEncoder(args)
        self.relation_path = args.relation_path
        self.absolute_path = args.absolute_path

        self.output_layer = nn.Linear(args.hidden, num_classes)

        self.embedding_para = torch.Tensor(4, self.args.lan_embedding_dim)
        nn.init.xavier_normal_(self.embedding_para)
        self.language_embedding_layer = nn.Parameter(self.embedding_para)

        self.language_embedding_projection = nn.Linear(self.args.lan_embedding_dim, self.args.projection_dim)

    def encode(self, data):
        content = data['content']
        content_mask = data['content_mask']
        path_map = data['path_map']
        paths = data['paths']
        paths_mask = data['paths_mask']
        r_paths = data['r_paths']
        r_paths_mask = data['r_paths_mask']
        r_path_idx = data['r_path_idx']
        named = data['named']
        language = data['language'].long()

        content_ = self.left_embedding(content, named)
        if self.relation_path:
            paths_ = self.path_embedding(paths, paths_mask, type='relation')
        else:
            paths_ = None
        if self.absolute_path:
            r_paths_ = self.path_embedding(r_paths, r_paths_mask, type='absolute')
        else:
            r_paths_ = None
        mask_ = (content_mask > 0).unsqueeze(1).repeat(1, content_mask.size(1), 1).unsqueeze(1)

        language_embedding = self.language_embedding_layer[language]
        language_embedding = self.language_embedding_projection(language_embedding)

        memory = self.encoder(content_, mask_, paths_, path_map, r_paths_, r_path_idx, language_embedding)

        return memory

    def forward(self, data):
        memory = self.encode(data)
        memory = torch.sum(memory, dim=1)
        out = self.output_layer(memory)
        return out


