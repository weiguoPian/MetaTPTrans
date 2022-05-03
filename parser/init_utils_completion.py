from tree_sitter import Language, Parser
import json
import os
from tqdm import tqdm
from .token_utils import split_func_name

root = os.getcwd()

def init_parser(language):
    # root = os.getcwd()
    # print(root)
    if language == 'multi':
        language_types = ['python', 'go', 'javascript', 'ruby']
        lang_parser_dict = dict()
        for lt in language_types:
            Language.build_library(
                '{}/build/{}.so'.format(root, language),
                [
                    '{}/vendor/tree-sitter-{}'.format(root, lt),
                ]
            )
            l = Language('{}/build/{}.so'.format(root, lt), lt)
            lang_parser = Parser()
            lang_parser.set_language(l)
            lang_parser_dict[lt] = lang_parser
        return lang_parser_dict
    else:
        Language.build_library(
            '{}/build/{}.so'.format(root, language),
            [
                '{}/vendor/tree-sitter-{}'.format(root, language),
            ]
        )
        language = Language('{}/build/{}.so'.format(root, language), language)
        lang_parser = Parser()
        lang_parser.set_language(language)
        return lang_parser


def node_dict_init(language):
    node_dic = dict()
    node_dic_path = os.path.join('{}/data/code_completion'.format(root), language, 'node_vocab.json')
    if os.path.exists(node_dic_path):
        with open(node_dic_path, 'r') as f:
            print('Already exist inter node dic')
            saved_node_dic = json.loads(f.readline())
            for key, value in saved_node_dic.items():
                node_dic[key] = value
    print(node_dic)
    return node_dic


def count_dict_init():
    count_dic = dict()
    keys = ['tokens', 'uni_paths', 'paths', 'named', 'func', 'nums', 'uni_r_paths', 'max_row', 'path_len']
    for k in keys:
        count_dic[k] = 0
    return count_dic


def read_files(language, type):
    file_list = []
    if language == 'multi':
        language_types = ['python', 'go', 'javascript', 'ruby']
        for lt in language_types:
            lt_dir_path = os.path.join('{}/raw_data/code_completion'.format(root), lt, type)
            lt_all_files = os.listdir(lt_dir_path)
            for file in lt_all_files:
                if 'gz' in file:
                    continue
                file = os.path.join(lt_dir_path, file)
                file_list.append(file)
    else:
        dir_path = os.path.join('{}/raw_data/code_completion'.format(root), language, type)
        all_files = os.listdir(dir_path)
        for file in all_files:
            if 'gz' in file:
                continue
            file_list.append(file)
    print(file_list)
    all_data = []
    for file in tqdm(file_list):
        if language == 'multi':
            f = open(file, 'r')
            # /raw_data/code_completion/python/train/python_train_0.jsonl
            lt = file.split('code_completion/')[-1].split('/')[0]
        else:
            f = open(os.path.join(dir_path, file), 'r')
        # with open(os.path.join(dir_path, file)) as f:
        lines = f.readlines()
        for line in lines:
            raw_data = json.loads(line)
            if language == 'multi':
                data = {'code': raw_data['code'], 'func_name': split_func_name(raw_data['func_name']), 'masked_code': raw_data['masked_code'], 'label': raw_data['label'], 'language': lt}
            else:
                data = {'code': raw_data['code'], 'func_name': split_func_name(raw_data['func_name']), 'masked_code': raw_data['masked_code'], 'label': raw_data['label']}
            all_data.append(data)
        f.close()
    print('Load {} {} files => {}'.format(language, type, len(all_data)))
    return all_data
