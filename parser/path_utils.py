from typing import List


def merge_terminals2_paths(v_path, u_path):
    s, n, m = 0, len(v_path), len(u_path)
    while s < min(n, m) and v_path[s] is u_path[s]:
        s += 1
    prefix, l_node = v_path[s:-1], v_path[-1]
    lca = v_path[s - 1]
    suffix, r_node = u_path[s:-1], u_path[-1]
    prefix = list(reversed(prefix))
    path = prefix + [lca] + suffix
    return l_node, prefix, [node.type for node in path], suffix, r_node


def save_path(path, pool):
    '''
    save path into path pool and return idx
    :param path:
    :param pool:
    :return:
    '''
    if len(pool) == 0:
        pool.append(path)
    else:
        for i, _lis in enumerate(pool):
            if _lis == path:
                return i
        pool.append(path)
    return len(pool) - 1


def paths_to_idx(paths, pool: List[List]) -> List:
    '''
    save a list of paths into path pool and get idx
    :param paths:
    :param pool:
    :return:
    '''
    path_idx = []
    for path in paths:
        path_ = [node.type for node in path[:-1]]
        path_idx.append(save_path(path_, pool))
    return path_idx


def path_convert(paths, node_dic) -> List[List]:
    '''
    convert word in paths to idx
    :param paths:
    :param node_dic:
    :return:
    '''

    def lookup(dic, item):
        if item not in dic:
            dic[item] = len(dic)
        return dic[item]

    temp_path = []
    for p in paths:
        temp_p = []
        for node in p:
            temp_p.append(lookup(node_dic, node))
        temp_path.append(temp_p)
    return temp_path
