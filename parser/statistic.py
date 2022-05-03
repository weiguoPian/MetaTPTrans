from typing import List, Dict


def token_statistic(source_dic: Dict, target_dic: Dict, source: List, target: List):
    def lookup_update(dic, item):
        if item in dic:
            dic[item] += 1
        else:
            dic[item] = 1
        return list(dic.keys()).index(item)

    for token in source:
        _ = lookup_update(source_dic, token)
    for token in target:
        _ = lookup_update(target_dic, token)


def data_count(data, count_dic):
    paths, code_tokens, code_named, func_name, paths_map, row, r_paths = \
        data['paths'], data['content'], data['named'], data['target'], data['paths_map'], data['row'], data['r_paths']
    count_dic['tokens'] += len(code_tokens)
    count_dic['uni_paths'] += len(paths)
    count_dic['paths'] += (sum([len(val) for key, val in paths_map.items()]) / 2 * 2)
    count_dic['named'] += code_named.count(1)
    count_dic['func'] += len(func_name)
    count_dic['nums'] += 1
    count_dic['uni_r_paths'] += (len(r_paths))
    count_dic['path_len'] += sum([len(i) for i in paths])
    if 'max_row' not in count_dic:
        count_dic['max_row'] = 0
    if max(row) > count_dic['max_row']:
        count_dic['max_row'] = max(row)

def data_count_completion(data, count_dic):
    paths, code_tokens, code_named, paths_map, row, r_paths = \
        data['paths'], data['content'], data['named'], data['paths_map'], data['row'], data['r_paths']
    count_dic['tokens'] += len(code_tokens)
    count_dic['uni_paths'] += len(paths)
    count_dic['paths'] += (sum([len(val) for key, val in paths_map.items()]) / 2 * 2)
    count_dic['named'] += code_named.count(1)
    # count_dic['func'] += len(func_name)
    count_dic['nums'] += 1
    count_dic['uni_r_paths'] += (len(r_paths))
    count_dic['path_len'] += sum([len(i) for i in paths])
    if 'max_row' not in count_dic:
        count_dic['max_row'] = 0
    if max(row) > count_dic['max_row']:
        count_dic['max_row'] = max(row)


def update_sum_dict(sub_source_dic, sub_target_dic, source_dic, target_dic):
    '''
    merge sub dic into whole dic
    :param sub_source_dic:
    :param sub_target_dic:
    :param source_dic:
    :param target_dic:
    :return:
    '''

    def update_dict(sub_dic, dic):
        for key, value in sub_dic.items():
            if key in dic:
                dic[key] += value
            else:
                dic[key] = value

    update_dict(sub_source_dic, source_dic)
    update_dict(sub_target_dic, target_dic)
