"""
Prepare Test Suit 0 and Test Suit 5 for IFT paper.
Last modified: 13980531
Author: Morteza ZAKERI
"""


import sys
import os
import subprocess
import re
import random
from random import shuffle
import shutil


def rename_files_in_directory():
    path = 'D:/afl_experiments_esxi_server/afl_queue_1000_top_file/'
    path = 'F:/outputs/queue/'
    for filename in os.listdir(path):
        new_filename = filename
        new_filename = new_filename.replace(':', '_')
        new_filename = new_filename.replace(',','-')
        shutil.move(path+filename, path+new_filename+'.pdf')
        print(filename)


def random_selcet_files_in_directory():
    source_path = 'E:/LSSDS/iust_pdf_corpus/corpus_merged_public/'
    destination_path = './TS0/'
    file_name_list = list()
    for filename in os.listdir(source_path):
        file_name_list.append(filename)
        print(filename)

    non_repeated_random_list = list()
    while len(non_repeated_random_list) < 1000:
        r = random.randint(0, 6140)
        if r not in non_repeated_random_list:
            non_repeated_random_list.append(r)
            print(r)
    print('Moving ...')
    for i in non_repeated_random_list:
        shutil.copy(source_path + file_name_list[i], destination_path)

# Main program to call above functions
def main(argv):
    # random_selcet_files_in_directory()
    rename_files_in_directory()


if __name__ == "__main__":
    main(sys.argv)
