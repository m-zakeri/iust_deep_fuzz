# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:13:22 2018
last edit:  1397-01-15
@author: Morteza
"""

__version__ = '0.2'
__author__ = 'Morteza'

import sys
import os
import subprocess
import random

import shutil

import pdf_object_extractor_2
from config import pdf_corpus_config


def get_stream_within_object(pdf_file_path=None,
                             mutool_path='D:\\afl\\mupdf-1.11-windows\\mutool.exe',
                             mutool_command=' show -b -e ',
                             mutool_object_number=' x'
                             ):
    """
    The core function of this script: get a single stream within the single pdf object
    :param pdf_file_path:
    :param mutool_path:
    :param mutool_command:
    :param mutool_object_number:
    :return:
    """
    cmd = mutool_path + mutool_command + pdf_file_path + ' ' + mutool_object_number
    returned_value_in_byte = subprocess.check_output(cmd, shell=True)

    # stream_object_id = 15
    # cmd = 'D:\\afl\\mupdf-1.11-windows\\mutool.exe show -b -e '
    # file = 'D:\\afl\\mupdf-1.11-windows\\input\\pdftc_100k_0001.pdf ' + str(stream_object_id)
    # cmd += file

    # return_value_in_string = returned_value_in_byte.decode()

    # can I use regexp?
    # return_value = re.sub(r'stream.*endstream', 'stream', return_value_in_string, flags=re.DOTALL)
    # print(return_value_in_string)

    # print(returned_value_in_byte)
    # x = return_value_in_string.replace('\r','-')
    # x = x.replace('\n','=')
    # print (x)

    return returned_value_in_byte


def write_stream_into_file(pdf_stream_byte=None, filename=None, obj_id=None):
    """

    :param pdf_stream_byte:
    :param filename:
    :param obj_id:
    :return:
    """
    stream_file_path = './selected_binary_streams/' + filename + str(obj_id).zfill(3)
    with open(stream_file_path, 'wb') as new_file:
        new_file.write(pdf_stream_byte)


def get_all_stream():
    """

    :return:
    """
    pdf_directory_path = pdf_corpus_config['corpus_merged']
    for filename in os.listdir(pdf_directory_path):
        try:
            print(pdf_directory_path+filename)
            start_index_integer, end_index_integer = \
                pdf_object_extractor_2.get_xref(pdf_file_path=pdf_directory_path+filename)
            objects_with_streams_id_list = []
            for i in range(start_index_integer, end_index_integer):
                pdf_object_string = \
                    pdf_object_extractor_2.get_pdf_object(pdf_file_path=pdf_directory_path+filename,
                                                          mutool_object_number=str(i))
                if pdf_object_string.find('stream') != -1:
                    objects_with_streams_id_list.append(str(i))
            for i, obj_id in enumerate(objects_with_streams_id_list):
                pdf_stram_byte = get_stream_within_object(pdf_file_path=pdf_directory_path+filename,
                                                          mutool_object_number=obj_id)
                write_stream_into_file(pdf_stram_byte, filename, obj_id)
                print('stream %s in file %s write successfully.' % (obj_id, filename))
        except Exception as e:
            print('Extracting from %s failed:' % filename, file=sys.stderr)
            print(str(e), file=sys.stderr)


def select_binary_streams_randomly(number=1000):
    stream_directory_path = pdf_corpus_config['corpus_merged_streams']
    filename_list = os.listdir(stream_directory_path)
    i = 0
    rand_prev = []
    while i < number:
        rand_curr = random.randint(0, len(filename_list)-1)
        while rand_curr in rand_prev:
            rand_curr = random.randint(0, len(filename_list)-1)
        rand_prev.append(rand_curr)
        src_path = pdf_corpus_config['corpus_merged_streams'] + filename_list[rand_curr]
        dst_path = './selected_binary_streams/' + filename_list[rand_curr]
        shutil.copyfile(src=src_path, dst=dst_path)
        i += 1
        print('file %s choose and copy.' % filename_list[rand_curr])
    print('total % file choose and copy' % number)


def main(argv):
    # get_all_stream()  # call this function on every pdf corpus and retrieve all binary stream within them.
    select_binary_streams_randomly(number=1000)


if __name__ == "__main__":
    main(sys.argv)
