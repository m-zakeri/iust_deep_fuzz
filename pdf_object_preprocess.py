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

# 01 - Remove all null and single / object from original trainset
def remove_null_and_slash_object(seq):
    seq = seq.replace('obj\nnull\nendobj', '')
    seq = seq.replace('obj\n\\\nendobj','')
    match = get_list_of_object(seq)
    print('number_of_not_null_or_slash_objects', len(match))
    save_to_file('./pdf_object_trainset_removed_null_and_slash.txt', seq)


# 02 - Remove first and last percentile (1/100) from dataset in step01
# number_of_not_null_or_slash_objects  is 494979
def remove_first_and_last_percentile(seq):
    match = get_list_of_object(seq, is_sort=True)
    # first percentile  = [0,4950]
    # last percentile = [490030, 494979]
    new_seq = ''
    count = 0
    for i in range(4951, 490031):
        new_seq += (match[i] + '\n')
        count += 1
    save_to_file('./pdf_object_trainset_removed_null_and_slash_and_percentile.txt', new_seq)
    print('number_of_objects', count)


# 02 - Create CSVs => Prepare for using in WEKA
def calculate_object_len_frequency(seq):
    match = get_list_of_object(seq)
    frequency_len_dic = {}
    len_list = list()
    count = 0
    for i, obj in enumerate(match):
        len_list.append(len(obj))
        frequency_len_dic.update({'obj'+str(i).zfill(6): len(obj)})
    c = Counter(len_list)
    # print(c)
    print(c.most_common(100))

    # with open('obj_len_frequencies.csv', 'w', newline='') as f:
    #     fieldnames = ['object_len', 'object_frequency']
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     data = [dict(zip(fieldnames, [k, v])) for k, v in c.items()]
    #     writer.writerows(data)

    with open('object_id__object_len_485080.csv', 'w', newline='') as f:
        fieldnames = ['object_id', 'object_len']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, v])) for k, v in frequency_len_dic.items()]
        writer.writerows(data)

    # with open("all_id_len.csv", 'w', newline='') as resultFile:
    #     for r in len_list:
    #         resultFile.write(str(r) + '\n')

    plt.plot(c.keys(), c.values(), 'go', label='line1')
    plt.show()


# 03 - Create iqr-cleaned dataset from csv step03 in WEKA i.e file
# 03_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_sorted.csv
def create_iqr_cleaned_dataset(seq):
    match = get_list_of_object(seq, is_sort=True)
    new_seq = ''
    count = 0
    for i in range(0, 477104):
        new_seq += (match[i] + '\n')
        count += 1
    save_to_file('./03_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_sorted_ii.txt', new_seq)
    print('number_of_objects', count)


# 04 - retrieve_trainset from seq step03 and csv step04
def retrieve_trainset(seq):
    """Read a CSV file using csv.DictReader"""
    csv_path = 'D:/iust_pdf_objects/preprocess/' \
               '04_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_testset_119276_shuffle.csv'
    object_id_list = []
    with open(csv_path, 'r') as f:
        # reader = csv.reader(f)
        reader = csv.DictReader(f, delimiter=',')
        for line in reader:
            # print(line["object_id"]),
            # print(line["object_len"])
            object_id_list.append(line['object_id'])
    print('end reading csv.')
    match = get_list_of_object(seq, is_sort=True)
    new_seq = ''
    for i, obj_id in enumerate(object_id_list):
        obj_id = int(obj_id[3:])
        new_seq += match[obj_id] + '\n\n'
    save_to_file('04_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_testset_119276_shuffle_ii.txt'
                 , new_seq)
    match = get_list_of_object(new_seq)
    print('trainset write successfully', len(match))


def statistical_analysis(seq):
    """ statistical analysis of corpus objects """
    match = get_list_of_object(seq)

    number_of_null_objects = seq.count('obj\nnull\nendobj')
    number_of_slash_objects = seq.count('obj\n\\\nendobj')
    number_of_stream_objects = seq.count('stream')

    # find min, max, ave len
    max_object_len = max([len(o) for o in match])
    min_object_len = min([len(o) for o in match])

    # sample_some_object_from_trainset(match)

    # print info
    print('number_of_total_objects: %s' % (len(match)))
    print('number_of_null_objects: %s' % number_of_null_objects)
    print('number_of_slash_objects: %s' % number_of_slash_objects)
    print('number_of_stream_objects: %s' % number_of_stream_objects)

    print('max_object_len %s' % max_object_len)
    print('min_object_len %s' % min_object_len)


# Bad way for sampling. I use it just for some test purposes.
def sample_some_object_from_trainset(match):
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

    print('ave_object_len %s' % (sum_of_lens / len(match)))
    print('number_of_total_characters %s' % sum_of_lens)
    print('number_of_separate_lens %s' % len(len_set))
    print('count ===', count)
    # save_to_file('./pdf_object_trainset_100_to_500_percent01.txt', seq_sort)


def main(argv):
    """comment"""
    # seq = concat()
    # path = 'D:/iust_pdf_objects/test_info.txt'
    # path = 'D:/iust_pdf_objects/preprocess/00_pdf_object_dataset_original_504153.txt'
    # path = 'D:/iust_pdf_objects/preprocess/01_pdf_object_dataset_removed_null_and_slash_494979.txt'
    path = 'D:/iust_pdf_objects/preprocess/02_pdf_object_dataset_removed_null_and_slash_and_percentile_485080_sorted.txt'
    path = 'D:/iust_pdf_objects/preprocess/03_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_sorted_ii.txt'
    seq = load_from_file(path)
    # remove_null_and_slash_object(seq)
    # remove_first_and_last_percentile(seq)
    # calculate_object_len_frequency(seq)
    # create_iqr_cleaned_dataset(seq)
    retrieve_trainset(seq)
    # statistical_analysis(seq)
    """end_of_main_function_script"""


if __name__ == "__main__":
    main(sys.argv)

