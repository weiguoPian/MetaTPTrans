def abs(length):
    return length if length >= 0 else 0


def convert_line(line):
    '''
    convert line into data dict
    :param line: 
    :return: 
    '''
    data = dict()
    # if type == 'multi':
    target, content, named, paths, paths_map, row, r_path_idx, r_paths, language = line.strip().split('\t')
    # else:
    #     target, content, named, paths, paths_map, row, r_path_idx, r_paths = line.strip().split('\t')
    data['target'] = target.split('|')
    data['content'] = content.split('|')
    data['named'] = [int(num) for num in named.split('|')]
    data['paths'] = [[int(num) for num in path.split()] for path in paths.split('|')]
    data['paths_map'] = [[int(num) for num in path_map.split()] for path_map in paths_map.split('|')]
    data['r_path_idx'] = [int(num) for num in r_path_idx.split('|')]
    data['r_paths'] = [[int(num) for num in r_path.split()] for r_path in r_paths.split('|')]
    data['row'] = [int(num) for num in row.split('|')]
    # if type == 'multi':
    data['language'] = language
    return data

def convert_line_completion(line):
    '''
    convert line into data dict
    :param line: 
    :return: 
    '''
    data = dict()
    # if type == 'multi':
    label, content, named, paths, paths_map, row, r_path_idx, r_paths, language = line.strip().split('\t')
    # else:
    #     label, content, named, paths, paths_map, row, r_path_idx, r_paths = line.strip().split('\t')
    data['label'] = label
    data['content'] = content.split('|')
    data['named'] = [int(num) for num in named.split('|')]
    data['paths'] = [[int(num) for num in path.split()] for path in paths.split('|')]
    data['paths_map'] = [[int(num) for num in path_map.split()] for path_map in paths_map.split('|')]
    data['r_path_idx'] = [int(num) for num in r_path_idx.split('|')]
    data['r_paths'] = [[int(num) for num in r_path.split()] for r_path in r_paths.split('|')]
    data['row'] = [int(num) for num in row.split('|')]
    # if type == 'multi':
    data['language'] = language
    return data


def decoder_process(target, vocab, max_target_len, e_voc=None, pointer=False):
    f_s = [vocab.find(sub_token) for sub_token in target]  # f_s not should been map, because need embedded
    if not pointer:
        f_t = [vocab.find(sub_token) for sub_token in target]
    else:
        assert e_voc is not None
        f_t = []
        for sub_token in target:
            if sub_token in e_voc:
                f_t.append(e_voc[sub_token])
            else:
                f_t.append(vocab.find(sub_token))
                # here still exist unk in f_t, cause perhaps some word not presented in method body
    f_source = [vocab.sos_index] + f_s
    f_source = f_source[:max_target_len] + [vocab.pad_index] * abs(max_target_len - len(f_source))
    # f_source: max_target_len
    f_target = f_t + [vocab.eos_index]
    f_target = f_target[:max_target_len] + [vocab.pad_index] * abs(max_target_len - len(f_target))
    # f_target: max_target_len

    return f_source, f_target


def row_process(row, max_code_length):
    min_row = min(row)
    row = [num - min_row + 1 for num in row]  # 0 for padding
    row_ = row[:max_code_length] + [0] * abs(max_code_length - len(row))
    return row_


def content_process(content, named, vocab, max_code_length, e_voc=None, pointer=False):
    content_ = [vocab.find(token) for token in content]
    if not pointer:
        content_e = None
    else:
        assert e_voc is not None
        content_e = []
        for sub_token in content:
            if sub_token in e_voc:
                content_e.append(e_voc[sub_token])
            else:
                content_e.append(vocab.find(sub_token))
    content_ = content_[:max_code_length] + [vocab.pad_index] * abs(max_code_length - len(content))
    if pointer:
        content_e = content_e[:max_code_length] + [vocab.pad_index] * abs(max_code_length - len(content))
    # content_: max_code_length
    content_mask_ = [1 for _ in content][:max_code_length] + [0] * abs(
        max_code_length - len(content))
    named_ = named[:max_code_length] + [2] * abs(max_code_length - len(named))  # 2 for padding
    return content_, content_mask_, named_, content_e


def path_process(paths, paths_map, max_path_num, max_code_length, path_embedding_num, max_path_length,
                 convert_hop=False):
    import torch
    # paths => # [[,,,,],[,,,,,]]
    # paths_map =>  # [[l,r],idx] => {idx:[l,r,l,r]}
    paths_map_ = [[max_path_num * 2 for _ in range(max_code_length)] for _ in
                  range(max_code_length)]  # we use <max_path_num>*2 to index the padding path
    paths_map_ = torch.tensor(paths_map_)

    # 1) use max_path_num to filter paths
    paths = paths[:max_path_num]
    # 2) use filtered paths and max_code_length to filter paths_map

    for key, value in enumerate(paths_map):
        assert len(value) % 2 == 0
        if int(key) >= max_path_num: break
        for i in range(0, len(value), 2):
            l, r = value[i], value[i + 1]
            if l >= max_code_length or r >= max_code_length:
                continue
            paths_map_[l, r] = int(key) * 2
            paths_map_[r, l] = int(key) * 2 + 1

    paths_mask_ = []
    paths_ = []
    for path in paths:
        if not convert_hop:
            paths_.append(
                path[:max_path_length] + [path_embedding_num] * abs(
                    max_path_length - len(path)))  # use path node num as padding idx of path
        else:
            paths_.append(
                [0] * min(max_path_length, len(path)) + [path_embedding_num] * abs(max_path_length - len(path)))
        paths_mask_.append(len(path) if len(path) < max_path_length else max_path_length)
        if not convert_hop:
            paths_.append(
                list(reversed(path))[:max_path_length] + [path_embedding_num] * abs(
                    max_path_length - len(path)))  # use path node num as padding idx of path
        else:
            paths_.append(
                [0] * min(max_path_length, len(path)) + [path_embedding_num] * abs(max_path_length - len(path)))
        paths_mask_.append(len(path) if len(path) < max_path_length else max_path_length)
        # reversed path for bidirectional gru

    assert len(paths_) <= max_path_num * 2
    assert len(paths_mask_) <= max_path_num * 2
    paths_ = paths_ + [[path_embedding_num] * max_path_length] * (
            max_path_num * 2 - len(paths_))
    paths_mask_ = paths_mask_ + [1] * (
            max_path_num * 2 - len(paths_mask_))  # 1 not 0 for padded path length
    return paths_map_, paths_, paths_mask_


def r_path_process(r_paths, r_path_idx, max_r_path_num, max_code_length, max_r_path_length, path_embedding_num,
                   convert_hop=False):
    # r_paths => # [[79,0],[...],[...]]
    # r_path_idx =>  # [0,1,1,1,3,,2,3,...]
    r_paths = r_paths[:max_r_path_num]
    r_path_idx_ = [idx if idx < max_r_path_num else max_r_path_num for idx in r_path_idx][
                  :max_code_length] + [max_r_path_num] * abs(
        max_code_length - len(r_path_idx))  # we use <max_r_path_num> to index the padding path
    r_paths_ = []
    r_paths_mask_ = []
    for r_path in r_paths:
        if not convert_hop:
            r_paths_.append(r_path[:max_r_path_length] + [path_embedding_num] * abs(max_r_path_length - len(r_path)))
        else:
            r_paths_.append(
                [0] * min(max_r_path_length, len(r_path)) + [path_embedding_num] * abs(max_r_path_length - len(r_path)))
        r_paths_mask_.append(
            len(r_path) if len(r_path) < max_r_path_length else max_r_path_length)
        if not convert_hop:
            r_paths_.append(list(reversed(r_path))[:max_r_path_length] +
                            [path_embedding_num] * abs(max_r_path_length - len(r_path)))
        else:
            r_paths_.append(
                [0] * min(max_r_path_length, len(r_path)) + [path_embedding_num] * abs(max_r_path_length - len(r_path)))
        r_paths_mask_.append(
            len(r_path) if len(r_path) < max_r_path_length else max_r_path_length)

    r_paths_ = r_paths_ + [[path_embedding_num] * max_r_path_length] * (
            max_r_path_num * 2 - len(r_paths_))
    r_paths_mask_ = r_paths_mask_ + [1] * (
            max_r_path_num * 2 - len(r_paths_mask_))
    return r_paths_, r_path_idx_, r_paths_mask_

def make_extended_vocabulary(content, vocab):
    e_voc, e_voc_ = dict(), dict()
    idx = len(vocab)
    for token in content:
        if not vocab.has_token(token) and token not in e_voc:
            e_voc[token] = idx
            e_voc_[idx] = token
            idx += 1
    return e_voc, e_voc_, idx
