from torch.utils.data import Dataset
import os
import torch
from .process_utils import convert_line_completion, decoder_process, row_process, content_process, path_process, r_path_process, \
    make_extended_vocabulary


class CompletionPathAttenDataset(Dataset):
    '''
    Please note for the dict keys presented in this file:

    The 'path' is the relative path, the 'paths_mask' is mask for padded relative path
    The 'r_path' means the path to 'r'oot, so it is the absolute path, and the 'r_paths_mask' is mask for padded absolute path

    The 'content' is the code tokens, the 'content_mask' is mask for token padding

    The 'path_map' is the matrix M for mapping, see appendix about the efficient computation of relative path encoding for details
    The 'r_path_idx' is used for reduce cost of absolute path, also see appendix about absolute path encoding

    The 'named' and 'row' are some additional structure information, but they are not much useful, so you can also ignore them
    '''

    def __init__(self, args, s_vocab, label_dict, type_):
        self.on_memory = args.on_memory
        self.dataset_dir = './data/code_completion/multi'
        self.s_vocab = s_vocab
        self.label_dict = label_dict
        self.args = args
        self.type_ = type_
        assert type_ in ['train', 'test', 'valid']
        if self.on_memory:
            self.json_path = os.path.join(self.dataset_dir, type_ + '.txt')
            with open(self.json_path, 'r') as f:
                self.data = f.readlines()
            self.corpus_line = len(self.data)
        else:
            self.json_path = os.path.join(self.dataset_dir, type_ + '.txt')
            self.corpus_line = 0
            with open(self.json_path, 'r') as f:
                for _ in f:
                    self.corpus_line += 1
            self.file = open(self.json_path, 'r')
        if self.args.tiny_data > 0:
            self.corpus_line = self.args.tiny_data
        self.hop = self.args.hop
        self.language2idx = {'python': 0, 'ruby': 1, 'javascript': 2, 'go': 3}


    def __len__(self):
        return self.corpus_line

    def __getitem__(self, item):
        assert item < self.corpus_line
        data = self.get_corpus_line(item)
        sample = self.process(data)
        res = {key: value if torch.is_tensor(value) or isinstance(value, dict) else torch.tensor(value) for key, value
                in sample.items()}
        # print(list(res.keys()))
        return res

    def process(self, data):
        label = self.label_dict[data['label']]
        row_ = row_process(data['row'], self.args.max_code_length)
        content_, content_mask_, named_, content_e = content_process(data['content'], data['named'], self.s_vocab,
                                                                     self.args.max_code_length)
        paths_map_, paths_, paths_mask_ = path_process(data['paths'], data['paths_map'], self.args.max_path_num,
                                                       self.args.max_code_length, self.args.path_embedding_num,
                                                       self.args.max_path_length, convert_hop=self.hop)
        r_paths_, r_path_idx_, r_paths_mask_ = r_path_process(data['r_paths'], data['r_path_idx'],
                                                              self.args.max_r_path_num,
                                                              self.args.max_code_length, self.args.max_r_path_length,
                                                              self.args.path_embedding_num, convert_hop=self.hop)
        data_dic = {'label': label, 'content': content_, 'content_mask': content_mask_,
                    'path_map': paths_map_, 'paths': paths_, 'paths_mask': paths_mask_, 'named': named_, 'row': row_,
                    'r_paths': r_paths_, 'r_path_idx': r_path_idx_, 'r_paths_mask': r_paths_mask_}

        data_dic['language'] = self.language2idx[data['language']]
        return data_dic

    def get_corpus_line(self, item):
        if self.on_memory:
            data = self.data[item]
            return convert_line_completion(data)
        else:
            if item == 0:
                self.file.close()
                self.file = open(self.json_path, 'r')
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.json_path, 'r')
                line = self.file.__next__()
            data = convert_line_completion(line)
            return data


def completion_collect_fn(batch):
    data = dict()
    max_content_len = 0
    for sample in batch:
        c_l = torch.count_nonzero(sample['content_mask']).item()
        if c_l > max_content_len: max_content_len = c_l
    data['label'] = torch.stack([b['label'] for b in batch])
    data['content'] = torch.stack([b['content'] for b in batch], dim=0)[:, :max_content_len]
    data['content_mask'] = torch.stack([b['content_mask'] for b in batch], dim=0)[:, :max_content_len]
    data['path_map'] = torch.stack([b['path_map'] for b in batch], dim=0)[:, :max_content_len, :max_content_len]
    data['paths'] = torch.stack([b['paths'] for b in batch], dim=0)
    data['paths_mask'] = torch.stack([b['paths_mask'] for b in batch], dim=0)
    data['named'] = torch.stack([b['named'] for b in batch], dim=0)[:, :max_content_len]
    data['row'] = torch.stack([b['row'] for b in batch], dim=0)[:, :max_content_len]
    data['r_paths'] = torch.stack([b['r_paths'] for b in batch], dim=0)
    data['r_path_idx'] = torch.stack([b['r_path_idx'] for b in batch], dim=0)[:, :max_content_len]
    data['r_paths_mask'] = torch.stack([b['r_paths_mask'] for b in batch], dim=0)
    if 'language' in batch[0]:
        data['language'] = torch.stack([b['language'] for b in batch])

    return data


def completion_collect_fn_inference(batch):
    data = dict()
    max_content_len = 512
    data['label'] = torch.stack([b['label'] for b in batch])
    data['content'] = torch.stack([b['content'] for b in batch], dim=0)[:, :max_content_len]
    data['content_mask'] = torch.stack([b['content_mask'] for b in batch], dim=0)[:, :max_content_len]
    data['path_map'] = torch.stack([b['path_map'] for b in batch], dim=0)[:, :max_content_len, :max_content_len]
    data['paths'] = torch.stack([b['paths'] for b in batch], dim=0)
    data['paths_mask'] = torch.stack([b['paths_mask'] for b in batch], dim=0)
    data['named'] = torch.stack([b['named'] for b in batch], dim=0)[:, :max_content_len]
    data['row'] = torch.stack([b['row'] for b in batch], dim=0)[:, :max_content_len]
    data['r_paths'] = torch.stack([b['r_paths'] for b in batch], dim=0)
    data['r_path_idx'] = torch.stack([b['r_path_idx'] for b in batch], dim=0)[:, :max_content_len]
    data['r_paths_mask'] = torch.stack([b['r_paths_mask'] for b in batch], dim=0)
    if 'language' in batch[0]:
        data['language'] = torch.stack([b['language'] for b in batch])

    return data