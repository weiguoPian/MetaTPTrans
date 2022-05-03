import argparse
import os
import itertools
from tqdm import tqdm
import attr
import random
from multiprocessing import Process
import json
import numpy as np
from .init_utils import init_parser, count_dict_init, node_dict_init, read_files
from .statistic import data_count, token_statistic, update_sum_dict
from .path_utils import path_convert, paths_to_idx, merge_terminals2_paths, save_path
from .token_utils import split_identifier_into_parts, is_number, is_punctuation, judge_func

identifier_type = {
    'python': ['identifier', 'list_splat_pattern', 'type_conversion'],
    'ruby': ['identifier', 'hash_key_symbol', 'simple_symbol', 'constant', 'instance_variable', 'global_variable',
             'class_variable'],
    'javascript': ['identifier', 'hash_key_symbol', 'simple_symbol', 'constant', 'instance_variable', 'global_variable',
                   'class_variable', 'property_identifier', 'shorthand_property_identifier', 'statement_identifier',
                   'shorthand_property_identifier_pattern', 'regex_flags'],
    'go': ['identifier', 'hash_key_symbol', 'simple_symbol', 'constant', 'instance_variable', 'global_variable',
           'class_variable', 'property_identifier', 'shorthand_property_identifier', 'statement_identifier',
           'shorthand_property_identifier_pattern', 'regex_flags', 'type_identifier', 'field_identifier',
           'package_identifier', 'label_name']
}
string_type = {
    'python': ['heredoc_content', 'string', 'comment', 'string_literal', 'character_literal', 'chained_string',
               'escape_sequence'],
    'ruby': ['heredoc_content', 'string', 'comment', 'string_literal', 'character_literal', 'chained_string',
             'escape_sequence', 'string_content', 'heredoc_beginning', 'heredoc_end'],
    'javascript': ['heredoc_content', 'string', 'comment', 'string_literal', 'character_literal', 'chained_string',
                   'escape_sequence', 'string_content', 'heredoc_beginning', 'heredoc_end', 'jsx_text',
                   'regex_pattern', 'string_fragment'],
    'go': ['heredoc_content', 'string', 'comment', 'string_literal', 'character_literal', 'chained_string',
           'escape_sequence', 'string_content', 'heredoc_beginning', 'heredoc_end', 'regex_pattern', '\n',
           'raw_string_literal', 'rune_literal']
}

root = os.getcwd()

@attr.s
class MyNode:
    type = attr.ib()
    named = attr.ib()
    idx = attr.ib()
    row = attr.ib()


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def clean_convert_split(args, paths, code, f_name, language):
    data_lines = code.splitlines()
    temp_paths = []
    func_name = []
    count = 0
    get_func = False
    for idx, path in enumerate(paths):
        if count >= args.max_code_length:
            break
        terminal = path[-1]
        l_, r_ = terminal.start_point, terminal.end_point
        if terminal.type in string_type[language]:
            new_node = MyNode('<{}>'.format('STR'), terminal.is_named, count, int(l_[0]))
            count += 1
            temp_paths.append(path + [new_node])
        else:
            # if l_[0] != r_[0]:
            #     print(terminal)
            assert l_[0] == r_[0]  # assert at same line
            literal = data_lines[l_[0]][l_[1]: r_[1]]
            blocks = split_identifier_into_parts(literal)
            if not get_func and judge_func(literal, f_name):  # this is func name
                func_name = blocks
                new_node = MyNode('<METHOD>', terminal.is_named, count, int(l_[0]))
                count += 1
                temp_paths.append(path + [new_node] if path[-1].type in identifier_type else path[:-1] + [new_node])
                get_func = True
            elif terminal.type in identifier_type[language]:
                for block in blocks:
                    new_node = MyNode(block, terminal.is_named, count, int(l_[0]))
                    count += 1
                    temp_paths.append(path + [new_node])
            elif is_number(literal) or terminal.type in ['decimal_integer_literal',
                                                         'decimal_floating_point_literal',
                                                         'hex_integer_literal', 'integer',
                                                         'float', 'int_literal', 'imaginary_literal', 'float_literal']:
                new_node = MyNode('<{}>'.format('NUM'), terminal.is_named, count, int(l_[0]))
                count += 1
                temp_paths.append(path + [new_node])
            elif not args.punctuation and is_punctuation(literal):
                continue
            else:
                new_node = MyNode(terminal.type, terminal.is_named, count, int(l_[0]))
                count += 1
                temp_paths.append(path[:-1] + [new_node])
    if not get_func:
        print(code)
    return temp_paths[:args.max_code_length], func_name


def language_parse(args, data, lang_parser):
    code, f_name = data['code'], data['func_name']
    if args.language == 'multi':
        lt = data['language']
        tree = lang_parser[lt].parse(bytes(code, "utf-8"))
    else:
        tree = lang_parser.parse(bytes(code, "utf-8"))
    stack, paths = [], []
    paths_map = dict()
    root_path_pool = []
    path_pool = []

    def dfs(node):
        stack.append(node)
        if len(paths) > args.max_code_length * 1.5:  # avoid no need dfs
            return
        if node.child_count == 0:
            paths.append(stack.copy())
        else:
            for child in node.children:
                dfs(child)
        stack.pop()

    cursor = tree.walk()
    dfs(cursor.node)
    if args.language == 'multi':
        paths, func_name = clean_convert_split(args, paths, code, f_name, data['language'])
    else:
        paths, func_name = clean_convert_split(args, paths, code, f_name, args.language)
    r_path_idx = paths_to_idx(paths, root_path_pool)
    terminals = [path[-1] for path in paths]
    combinations = itertools.combinations(iterable=paths, r=2)
    for v_path, u_path in combinations:
        l_node, prefix, path, suffix, r_node = merge_terminals2_paths(v_path, u_path)
        if len(path) > args.max_path_length:
            idx = np.linspace(0, len(path) - 1, args.max_path_length, dtype=int).tolist()
            path = [path[i] for i in idx]
        assert len(path) <= args.max_path_length
        path_idx = save_path(path, path_pool)
        if path_idx in paths_map:
            paths_map[path_idx].append(terminals.index(l_node))
            paths_map[path_idx].append(terminals.index(r_node))
        else:
            paths_map[path_idx] = [terminals.index(l_node), terminals.index(r_node)]
    return path_pool, [node.type for node in terminals], [int(node.named) for node in terminals], func_name, \
           paths_map, [int(node.row) + 1 for node in terminals], r_path_idx, root_path_pool


def sub_process(args, idx, all_data, lang_parser):
    save_path = os.path.join('{}/data/summarization'.format(root), args.language, '{}_{}.json'.format(args.type, idx))
    source_dic, target_dic = dict(), dict()
    with open(save_path, 'w') as f:
        for data in tqdm(all_data):
            f_name = data['func_name']
            if len(f_name) == 0:  # javascript anonymous function should been removed, following code transformer
                continue
            paths, code_tokens, code_named, func_name, paths_map, row, r_path_idx, r_paths = \
                language_parse(args, data, lang_parser)
            token_statistic(source_dic, target_dic, code_tokens, func_name)
            if args.language == 'multi':
                data = {'target': func_name,
                        'content': code_tokens, 'named': code_named,
                        'paths': paths, 'paths_map': paths_map, 'row': row, 'r_path_idx': r_path_idx, 'r_paths': r_paths, 'language': data['language']
                        }
            else:
                data = {'target': func_name,
                        'content': code_tokens, 'named': code_named,
                        'paths': paths, 'paths_map': paths_map, 'row': row, 'r_path_idx': r_path_idx, 'r_paths': r_paths
                        }
            f.write(json.dumps(data) + '\n')
    dict_save_path = os.path.join('{}/data/summarization'.format(root), args.language, '{}_dict_{}.json'.format(args.type, idx))
    with open(dict_save_path, 'w') as f:
        f.write(json.dumps(source_dic) + '\n')
        f.write(json.dumps(target_dic) + '\n')


def compress(args, data):
    # if args.language == 'multi':
    #     target, content, named, paths, paths_map, r_path_idx, r_paths, row, lt = \
    #         data['target'], data['content'], data['named'], data['paths'], data['paths_map'], data[
    #             'r_path_idx'], data['r_paths'], data['row'], data['language']

    #     s = "|".join([word for word in target]) + '\t' + "|".join([word for word in content]) + '\t' \
    #         + "|".join([str(num) for num in named]) + '\t' + \
    #         "|".join([" ".join([str(num) for num in path]) for path in paths]) + '\t' + \
    #         "|".join([" ".join([str(num) for num in value]) for key, value in paths_map.items()]) + '\t' + "|".join(
    #         [str(num) for num in row]) + '\t' + "|".join([str(num) for num in r_path_idx]) + '\t' + \
    #         "|".join([" ".join([str(num) for num in r_path]) for r_path in r_paths]) + '\t' + lt

    # else:
    target, content, named, paths, paths_map, r_path_idx, r_paths, row = \
        data['target'], data['content'], data['named'], data['paths'], data['paths_map'], data[
            'r_path_idx'], data['r_paths'], data['row']

    s = "|".join([word for word in target]) + '\t' + "|".join([word for word in content]) + '\t' \
        + "|".join([str(num) for num in named]) + '\t' + \
        "|".join([" ".join([str(num) for num in path]) for path in paths]) + '\t' + \
        "|".join([" ".join([str(num) for num in value]) for key, value in paths_map.items()]) + '\t' + "|".join(
        [str(num) for num in row]) + '\t' + "|".join([str(num) for num in r_path_idx]) + '\t' + \
        "|".join([" ".join([str(num) for num in r_path]) for r_path in r_paths])
    
    if args.language == 'multi':
        s = s + '\t' + data['language']

    return s


def process(args):
    if args.language == 'multi':
        lang_parser_dict = init_parser(args.language)
    else:
        lang_parser = init_parser(args.language)
    if not os.path.exists('{}/data/summarization/{}'.format(root, args.language)):
        os.makedirs('{}/data/summarization/{}'.format(root, args.language))
    source_dic, target_dic = dict(), dict()
    node_dic = node_dict_init(args.language)
    count_dic = count_dict_init()
    all_data = read_files(args.language, args.type)
    if args.shuffle: random.shuffle(all_data)
    if args.nums > 0: all_data = all_data[:args.nums]
    if args.process_num > 1:
        pool = []
        split_data = [[] for _ in range(args.process_num)]
        for i in range(len(all_data)):
            split_data[i % args.process_num].append(all_data[i])
        for i in range(args.process_num):
            if args.language == 'multi':
                pool.append(Process(target=sub_process, args=(args, i, split_data[i], lang_parser_dict)))
            else:
                pool.append(Process(target=sub_process, args=(args, i, split_data[i], lang_parser)))
            pool[-1].start()
        for p in pool:
            p.join()
    else:
        if args.language == 'multi':
            sub_process(args, 0, all_data, lang_parser_dict)
        else:
            sub_process(args, 0, all_data, lang_parser)

    print('Sub Files Merge')
    sum_save_path = os.path.join('{}/data/summarization'.format(root), args.language, '{}.txt'.format(args.type))
    with open(sum_save_path, 'w') as f:
        for i in range(args.process_num):
            sub_save_path = os.path.join('{}/data/summarization'.format(root), args.language, '{}_{}.json'.format(args.type, i))
            with open(sub_save_path, 'r') as l:
                lines = l.readlines()
                for line in lines:
                    data = json.loads(line)
                    data_count(data, count_dic)
                    data['paths'] = path_convert(data['paths'], node_dic)
                    data['r_paths'] = path_convert(data['r_paths'], node_dic)
                    f.write(compress(args, data) + '\n')
            os.remove(sub_save_path)

    print('Sub Dict Concat')
    for i in range(args.process_num):
        sub_dict_path = os.path.join('{}/data/summarization'.format(root), args.language, '{}_dict_{}.json'.format(args.type, i))
        try:
            with open(sub_dict_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2
                sub_source_dic = json.loads(lines[0])
                sub_target_dic = json.loads(lines[1])
                update_sum_dict(sub_source_dic, sub_target_dic, source_dic, target_dic)
            os.remove(sub_dict_path)
        except FileNotFoundError:
            continue

    print('avg_tokens:{}'.format(count_dic['tokens'] / count_dic['nums']))
    print('avg_comment_tokens:{}'.format(count_dic['func'] / count_dic['nums']))
    print('avg_uni_undirected_paths:{}'.format(count_dic['uni_paths'] / count_dic['nums']))
    print('avg_directed_paths:{}'.format(count_dic['paths'] / count_dic['nums']))
    print('avg_path_len:{}'.format(count_dic['path_len'] / count_dic['uni_paths']))
    print('avg_uni_r_paths:{}'.format(count_dic['uni_r_paths'] / count_dic['nums']))
    print('named_ratio:{}'.format(count_dic['named'] / count_dic['tokens']))
    print('path_ratio:{}'.format((count_dic['paths'] / count_dic['nums']) / (
            (count_dic['tokens'] / count_dic['nums']) * (count_dic['tokens'] / count_dic['nums'] - 1) / 2)))
    print('max_row:{}'.format(count_dic['max_row']))
    print('source_vocab:{}'.format(len(source_dic)))
    print('target_vocab:{}'.format(len(target_dic)))
    print('node_vocab:{}'.format(len(node_dic)))
    if args.type == 'train' or args.save_vocab:
        print('Save Text Vocab')
        with open('{}/data/summarization/{}/source_vocab.json'.format(root, args.language), 'w') as f:
            json.dump(source_dic, f)
        with open('{}/data/summarization/{}/target_vocab.json'.format(root, args.language), 'w') as f:
            json.dump(target_dic, f)
    print('Save Node Vocab')
    with open('{}/data/summarization/{}/node_vocab.json'.format(root, args.language), 'w') as f:
        json.dump(node_dic, f)


if __name__ == '__main__':
    # print(root)
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['python', 'javascript', 'ruby', 'go', 'multi'], type=str, default='javascript')
    parser.add_argument('--type', choices=['train', 'valid', 'test', 'temp'], type=str, default='test')
    parser.add_argument('--max_path_length', type=int, default=32)
    parser.add_argument('--max_code_length', type=int, default=512)
    parser.add_argument('--nums', type=int, default=-1)
    parser.add_argument('--punctuation', type=boolean_string, default=False)
    parser.add_argument('--process_num', type=int, default=16)
    parser.add_argument('--save_vocab', type=boolean_string, default=False)
    parser.add_argument('--shuffle', type=boolean_string, default=False)
    args = parser.parse_args()
    print(args)
    process(args)
