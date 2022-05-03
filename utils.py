from asyncore import write
from cgi import print_arguments
from opcode import opname
import os
import json
import re
from turtle import goto

from numpy import source

from parser.multi_language_parser import language_parse

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


def marge_ct_vocab(language_types):
    res = dict()
    for t in language_types:
        ct_vocab_path = './data/{}/ct_vocab.json'.format(t)
        with open(ct_vocab_path, 'r') as load_f:
            load_dict = json.load(load_f)
        print('{}: {}'.format(t, len(load_dict)))
        for key in load_dict:
            if key not in res:
                res[key] = len(res)
    
    print(len(res))
    
    res_path = './data/multi/ct_vocab.json'.format(t)
    with open(res_path, 'w') as save_f:
        json.dump(res, save_f)


# def update_identifier_type_dict(language_types):
#     identifier_type['multi'] = []
#     for lt in language_types:
#         for identifier in identifier_type[lt]:
#             if identifier not in identifier_type['multi']:


def checkTxt_1():
    file_path = './data/code_completion/ruby/test.txt'
    f = open(file_path, 'r')
    num = 0
    fault_num = 0
    for line in f:
        num += 1
        if num % 1000 == 0:
            print(num)
        if len(line.strip().split('\t')) < 8:
            # print()
            fault_num += 1
    f.close()
    print(fault_num)

def checkTxt_2():
    file_path = './data/multi/test.txt'
    lan_num_dict = dict()
    lan_num_dict['python'] = 0
    lan_num_dict['go'] = 0
    lan_num_dict['javascript'] = 0
    lan_num_dict['ruby'] = 0
    f = open(file_path, 'r')
    num = 0
    for line in f:
        num += 1
        if num % 1000 == 0:
            print(num)
        lan = line.strip().split('\t')[-1]
        lan_num_dict[lan] += 1
        # break
    f.close()
    print(lan_num_dict)

def new_txt():
    ori_file_path = './data/javascript/ori_train.txt'
    target_file_path = './data/javascript/train.txt'
    ori_f = open(ori_file_path, 'r')
    tar_f = open(target_file_path, 'w')
    # num = 0
    num = 0
    for line in ori_f:
        num += 1
        if num % 1000 == 0:
            print(num)
        if len(line.strip().split('\t')) < 8:
            continue
        tar_f.write(line)
        # num += 1
        # break
    ori_f.close()
    tar_f.close()
    # print(num)

def length(lan='go'):
    train_path = './raw_data/{}/train'.format(lan)
    valid_path = './raw_data/{}/valid'.format(lan)
    test_path = './raw_data/{}/test'.format(lan)
    num = 0
    total_num = 0
    for jf_name in os.listdir(train_path):
        jf = os.path.join(train_path, jf_name)
        with open(jf, 'r') as f:
            for line in f:
                total_num += 1
                line_dict = json.loads(line)
                token_list = line_dict['code_tokens']
                if len(token_list) < 100:
                    num += 1
    print(num)
    print(total_num)

def lens():
    go_path = 'data/go/source_vocab.json'
    python_path = 'data/python/source_vocab.json'
    javascript_path = 'data/javascript/source_vocab.json'
    ruby_path = 'data/ruby/source_vocab.json'
    multi_path = 'data/multi/source_vocab.json'

    # path = 'data/multi/ct_vocab.json'
    num = 0
    total_num = 0
    with open(multi_path, 'r') as f:
        j = json.load(f)
        print(len(j))
        for key in j:
            total_num += j[key]
            # total_num += 1
            # if j[key] >= 100:
            #     num += 1
    # print(num)
    print(total_num)

def token_percent():
    path = 'data/multi/source_vocab.json'
    total = 0
    valid = 0
    throw_out = 0
    with open(path, 'r') as f:
        j = json.load(f)
        for key in j:
            total += j[key]
            if j[key] >= 100:
                valid += j[key]
            else:
                throw_out += j[key]
    print(total)
    print(valid)
    print(throw_out)
    print(valid/total)
    print(throw_out/total)

def check_completion_data():
    path = './data/code_completion/python/train.txt'
    f = open(path, 'r')
    num = 0
    # fault_num = 0
    for line in f:
        num += 1
        if num == 10:
            break
        l = line.strip().split('\t')[0]
        print(l)
        # break
        # num += 1
        # if num % 1000 == 0:
        #     print(num)
        # if len(line.strip().split('\t')) < 8:
        #     # print()
        #     fault_num += 1
    f.close()
    # print(fault_num)

def isExistMask():
    root = './raw_data/python/train'
    list_jsl = os.listdir(root)
    count = 0
    # has = None
    for jsl in list_jsl:
        print(jsl)
        path = os.path.join(root, jsl)
        f = open(path, 'r')
        num = 0
        for line in f:
            # num += 1
            # if num % 1000 == 0:
            #     print('{}: {}'.format(jsl, num))
            line_data = json.loads(line)
            code = line_data['code']
            if '[MASK]' in code:
                print(code)
                count += 1
    print()
    print(count)

def refine_source_vocab(lan):
    # lan = 'multi'
    label_vocab_path = './data/code_completion/label_vocab.json'
    old_source_vocab_path = './data/code_completion/{}/source_vocab.json'.format(lan)
    refined_source_vocab_path = './data/code_completion/{}/refined_source_vocab.json'.format(lan)

    label_vocab = open(label_vocab_path, 'r')
    label_vocab_dict = json.load(label_vocab)

    old_source_vocab = open(old_source_vocab_path, 'r')
    old_source_vocab_dict = json.load(old_source_vocab)

    refined_source_vocab = open(refined_source_vocab_path, 'w')
    # refined_source_vocab_dict = json.load(refined_source_vocab)

    for key in label_vocab_dict:
        if key not in old_source_vocab_dict:
            old_source_vocab_dict[key] = label_vocab_dict[key]
        else:
            old_source_vocab_dict[key] += label_vocab_dict[key]
    
    refined_source_vocab = open(refined_source_vocab_path, 'w')
    refined_source_vocab.write(json.dumps(old_source_vocab_dict))

    label_vocab.close()
    old_source_vocab.close()
    refined_source_vocab.close()


def testRefine():
    lan = 'multi'
    old_path = './data/code_completion/{}/old_source_vocab.json'.format(lan)
    new_path = './data/code_completion/{}/source_vocab.json'.format(lan)

    with open(old_path, 'r') as old_f:
        old_dict = json.load(old_f)
    with open(new_path, 'r') as new_f:
        new_dict = json.load(new_f)
    
    old_num = 0
    for key in old_dict:
        old_num += old_dict[key]
    
    new_num = 0
    for key in new_dict:
        new_num += new_dict[key]
    
    print(old_num)
    print(new_num)
    print(len(old_dict))
    print(len(new_dict))

def combine_label_vocab():
    lans = ['python', 'go', 'javascript', 'ruby']
    root = './data/code_completion'

    target_path = os.path.join(root, 'label_vocab.json')
    target_f = open(target_path, 'w')
    new_dict = dict()

    for lan in lans:
        print(lan)
        lan_label_vocab_path = os.path.join(root, lan, 'label_vocab.json')
        lan_label_vocab_f = open(lan_label_vocab_path, 'r')
        lan_label_dict = json.load(lan_label_vocab_f)
        for key in lan_label_dict:
            if key not in new_dict:
                new_dict[key] = lan_label_dict[key]
            else:
                new_dict[key] += lan_label_dict[key]
        lan_label_vocab_f.close()
    target_f.write(json.dumps(new_dict))
    target_f.close()

def checkNewLabelVocab():
    path = './data/code_completion/label_vocab.json'
    with open(path, 'r') as f:
        d = json.load(f)
    print(d)
    print(len(d))

def label_vocab_to_label_dict():
    lan = 'ruby'
    label_vocab_path = './data/code_completion/{}/label_vocab.json'.format(lan)
    label_dict_path = './data/code_completion/{}/label_dict.json'.format(lan)
    label_vocab_f = open(label_vocab_path, 'r')
    label_vocab = json.load(label_vocab_f)
    label_vocab_f.close()
    new_dict = dict()

    num = 0
    for key in label_vocab:
        new_dict[key] = num
        num += 1

    label_dict_f = open(label_dict_path, 'w')
    label_dict_f.write(json.dumps(new_dict))
    label_dict_f.close()


if __name__ == '__main__':
    language_types = ['python', 'go', 'javascript', 'ruby']
    # marge_ct_vocab(language_types)
    # checkTxt_1()
    # new_txt()
    # f = open('./data/javascript/train.txt')
    # for line in f:
    #     print(line.strip().split('\t'))
    #     print(len(line.strip().split('\t')))
    #     break
    # f.close()
    # length(lan='ruby')
    # lens()
    # token_percent()
    # check_completion_data()
    # isExistMask()
    # refine_source_vocab()
    # testRefine()
    # lens()
    # combine_label_vocab()
    # checkNewLabelVocab()
    # label_vocab_to_label_dict()
    refine_source_vocab(lan='python')