# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:13:22 2018

@author: Morteza
"""

import sys
import subprocess
import re
import os

from collections import Counter
import csv
import matplotlib.pyplot as plt


def save_to_file(path, seq):
    with open(path, 'w', encoding='utf8') as cf:
        cf.write(seq)


def load_from_file(path):
    with open(path, mode='r', encoding='utf8') as cf:
        seq = cf.read()
        return seq


def concat():
    pdf_object_directory_path = 'D:/iust_pdf_objects/all_object_files_514190_object_6101_out_of_6163_file/'
    concat_seq = ''
    for filename in os.listdir(pdf_object_directory_path):
        with open(pdf_object_directory_path + filename, 'r') as f:
            concat_seq += f.read()
            # concat_seq += '\n' + ('-' * 75) + '\n'
        print(filename, 'read successfully')
    # concat_seq = concat_seq.replace('\n\n', '\n')
    concat_file = 'D:/iust_pdf_objects/pdf_object_trainset_with_devider.txt'
    concat_seq = sanitize(concat_seq)
    # save_to_file(concat_file,concat_seq)
    print('end successfully')
    return concat_seq


def sanitize(seq):
    """sanitize concat sequence"""
    x = seq.count('obj')
    print(x)
    # seq =
    return seq


def get_list_of_object(seq, is_sort=True):
    """ return a sorted list of all data object in the seq (but why sort?) """
    # regexp to match all type of objects exist in sequence
    regex = re.compile(r'obj\b(?:(?!endobj).)*\bendobj', flags=re.DOTALL)
    # match is now a list contains all object in a seq string
    match = regex.findall(seq)

    if is_sort:
        # sort match by len of objects
        match.sort(key=lambda s: len(s))
    return match


def statistical_analysis(seq):
    """ statistical analysis of corpus objects """
    match = get_list_of_object(seq)

    number_of_null_objects = seq.count('obj\nnull\nendobj')
    number_of_slash_objects = seq.count('obj\n\\\nendobj')
    number_of_stream_objects = seq.count('stream')

    # find min, max, ave len
    max_object_len = max([len(o) for o in match])
    min_object_len = min([len(o) for o in match])

    sum_of_lens = 0
    seq_sort = ''
    len_set = set()
    count = 0
    for i, o in enumerate(match):
        sum_of_lens += len(o)
        print(len(o))
        len_set.add(len(o))
        if (25 < len(o) < 500) and ((i % 100) == 0):
            seq_sort += str(o) + '\n'
            count += 1

    # save_to_file('D:/iust_pdf_objects/pdf_object_trainset_100_to_500_percent01.txt', seq_sort)

    # print info
    print('number_of_total_objects: %s' % (len(match)))
    print('number_of_null_objects: %s' % number_of_null_objects)
    print('number_of_slash_objects: %s' % number_of_slash_objects)
    print('number_of_stream_objects: %s' % number_of_stream_objects)

    print('max_object_len %s' % max_object_len)
    print('min_object_len %s' % min_object_len)
    print('ave_object_len %s' % (sum_of_lens / len(match)))
    print('number_of_total_characters %s' % sum_of_lens)

    print('number_of_separate_lens %s' % len(len_set))

    print('count ===', count)


def calculate_object_len_frequency(seq):
    match = get_list_of_object(seq)
    frequency_len_dic = {}
    len_list = list()
    count = 0
    for i, obj in enumerate(match):
        len_list.append(len(obj))
        frequency_len_dic.update({'obj'+str(i).zfill(6):len(obj)})
    c = Counter(len_list)
    # print(c)
    print(c.most_common(100))

    # with open('obj_len_frequencies.csv', 'w', newline='') as f:
    #     fieldnames = ['object_len', 'object_frequency']
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     data = [dict(zip(fieldnames, [k, v])) for k, v in c.items()]
    #     writer.writerows(data)

    with open('obj_id_len.csv', 'w', newline='') as f:
        fieldnames = ['object_id', 'object_len']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, v])) for k, v in frequency_len_dic.items()]
        writer.writerows(data)


    # with open("all_id_len.csv", 'w', newline='') as resultFile:
    #     for r in len_list:
    #         resultFile.write(str(r) + '\n')

    # plt.plot(c.keys(), c.values(), 'go', label='line1')
    # plt.show()


def main(argv):
    """comment"""
    # seq = concat()
    # path = 'D:/iust_pdf_objects/test_info.txt'
    path = 'D:/iust_pdf_objects/pdf_object_trainset.txt'
    seq = load_from_file(path)
    # statistical_analysis(seq)
    calculate_object_len_frequency(seq)
    """comment"""


if __name__ == "__main__":
    main(sys.argv)
