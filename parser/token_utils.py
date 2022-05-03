import string
from typing import List
import re


def is_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def is_punctuation(s):

    for s_ in s:
        if s_ not in string.punctuation:
            return False
    return True


def split_camelcase(camel_case_identifier: str) -> List[str]:
    """
    Split camelCase identifiers.
    come from code transformer
    """
    if not len(camel_case_identifier):
        return []
    # split into words based on adjacent cases being the same
    result = []
    current = str(camel_case_identifier[0])
    prev_upper = camel_case_identifier[0].isupper()
    prev_digit = camel_case_identifier[0].isdigit()
    prev_special = not camel_case_identifier[0].isalnum()
    for c in camel_case_identifier[1:]:
        upper = c.isupper()
        digit = c.isdigit()
        special = not c.isalnum()
        new_upper_word = upper and not prev_upper
        new_digit_word = digit and not prev_digit
        new_special_word = special and not prev_special
        if new_digit_word or new_upper_word or new_special_word:
            result.append(current)
            current = c
        elif not upper and prev_upper and len(current) > 1:
            result.append(current[:-1])
            current = current[-1] + c
        elif not digit and prev_digit:
            result.append(current)
            current = c
        elif not special and prev_special:
            result.append(current)
            current = c
        else:
            current += c
        prev_digit = digit
        prev_upper = upper
        prev_special = special
    result.append(current)
    return result


def split_identifier_into_parts(identifier: str) -> List[str]:
    """
    Split a single identifier into parts on snake_case and camelCase
    come from code transformer
    """
    # if identifier == '<MASK>':
    #     return [identifier]
    snake_case = identifier.split("_")

    identifier_parts = []  # type: List[str]
    for i in range(len(snake_case)):
        part = snake_case[i]
        if len(part) > 0:
            identifier_parts.extend(s.lower() for s in split_camelcase(part))
    if len(identifier_parts) == 0:
        return [identifier]
    return identifier_parts


def split_word(word):
    '''
    old word split method
    :param word:
    :return:
    '''
    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]

    blocks = []
    for underscore_block in word.split('_'):
        blocks.extend(camel_case_split(underscore_block))
    f = []
    for block in blocks:
        t = re.sub("[^A-Za-z]", "", block)
        if len(t) != 0:
            f.append(t)
    if len(f) == 0:
        f = blocks if len(blocks) != 0 else ['_']
    return [block.lower() for block in f]


def split_func_name(func_name):
    return func_name[func_name.rindex('.') + 1:] if '.' in func_name else func_name


def judge_func(source, target):
    '''
    value=   <==>     value
    tree sitter will remove = after value, so cant find func name
    :param source:
    :param target:
    :return:
    '''
    if source == target:
        return True
    s, t = split_word(source), split_word(target)
    if s == t:
        return True
    return False


if __name__ == '__main__':
    func_name = 'Infinity,'
    print(split_identifier_into_parts(func_name))

