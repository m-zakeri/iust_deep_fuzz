# -*- coding: utf-8 -*-
"""
File name: pdf_object_extractor_2.py
Created on Mon Jan  8 15:13:22 2018 (1396-10-16)
Last update on 1397-01-14
@author: Morteza Zakeri

Description:
Version 2 of pdf_pdf_object_extractor.py
Version 2 new change set:
 - The code is now more portable

Version 1:
This file use to extract data object in (set of) PDF files.
We use mutool a subprogram of mupdf an open source pdf library, readers and tools.

"""

import sys
import subprocess
import re
import os


# change mutool_path to point the location of mupdf
def get_xref(pdf_file_path=None,
             mutool_path='D:/afl/mupdf-1.11-windows/mutool.exe',
             mutool_command=' show -e ',
             mutool_object_number=' x'):
    """
    command line to get xref size
    :param pdf_file_path:
    :param mutool_path:
    :param mutool_command:
    :param mutool_object_number:
    :return:
    """
    cmd = mutool_path + mutool_command + pdf_file_path + ' ' + mutool_object_number
    # cmd = 'D:\\afl\\mupdf-1.11-windows\\mutool.exe show -e  D:\\afl\\mupdf-1.11-windows\\input\\pdftc_100k_2708.pdf x'

    returned_value_in_byte = subprocess.check_output(cmd, shell=True)
    return_value_in_string = returned_value_in_byte.decode()
    # print command line output to see it in python console
    # print(return_value_in_string)
    start_index_string = ''
    end_index_string = ''
    index = 6
    while(return_value_in_string[index].isdigit()):
        start_index_string += return_value_in_string[index]
        index += 1
    index += 1
    while(return_value_in_string[index].isdigit()):
        end_index_string += return_value_in_string[index]
        index += 1

    start_index_integer = int(start_index_string)
    end_index_integer = int(end_index_string)
    # print(start_index_integer, end_index_integer)
    return start_index_integer,end_index_integer


def get_pdf_object(pdf_file_path=None,
                   mutool_path='D:/afl/mupdf-1.11-windows/mutool.exe',
                   mutool_command=' show -e ',
                   mutool_object_number=' x'):
    """ get a single object with id ' x'
    """
    cmd = mutool_path + mutool_command + pdf_file_path + ' ' + mutool_object_number
    # Execute the cmd command and return output of command e.g. pdf objects
    returned_value_in_byte = subprocess.check_output(cmd, shell=True)
    # Convert output to string
    return_value_in_string = returned_value_in_byte.decode()
    return return_value_in_string


def get_pdf_objects(pdf_file_path=None,
                    mutool_path='D:\\afl\\mupdf-1.11-windows\\mutool.exe',
                    mutool_command=' show -e ',
                    mutool_object_number=' x'):
    """ get all object with id ' x1, x2, x3, ..., xn'"""
    cmd = mutool_path + mutool_command + pdf_file_path + mutool_object_number
    #cmd = 'D:\\afl\\mupdf-1.11-windows\\mutool.exe show -e  D:\\afl\\mupdf-1.11-windows\\input\\pdftc_100k_2708.pdf '
    # Execute the cmd command and return output of command e.g. pdf objects
    returned_value_in_byte = subprocess.check_output(cmd, shell=True)
    # Convert output to string
    return_value_in_string = returned_value_in_byte.decode()


    # Below we sanitize the output and just keep text in new_seq variable (4 total steps)
    new_seq = ''

    # Can I use regexp for sanitize? It seems not yet!
    #return_value = re.sub(r'stream.*endstream', 'stream', return_value_in_string, flags=re.DOTALL)
    #print(return_value_in_string)

    # regexp to match all streams exist in sequence
    # regex = re.compile(r'stream\b(?:(?!endstream).)*\bendstream', flags=re.DOTALL)
    # match = regex.findall(return_value_in_string)


    # Step 1: Eliminate all binary stream within the data object body by using below loop instead of above regexp
    stream_start = 0
    stream_end = 0
    while return_value_in_string.find('endstream', stream_end+9) != -1:
        stream_start = return_value_in_string.find('stream', stream_end+9)
        new_seq += return_value_in_string[stream_end:stream_start+6]
        stream_end = return_value_in_string.find('endstream', stream_end+9)
    #   if(stream_start != -1 and stream_end != -1):
    #       for i in range(stream_start+6,stream_end):
    #           return_value_in_list.(i)
    #   print(stream_start, stream_end)
    new_seq += return_value_in_string[stream_end:len(return_value_in_string)]

    # Step 2: Mark the places that exist a binary stream with keyword stream and put it in the final output
    new_seq = new_seq.replace('streamendstream', 'stream')
    #print(new_seq)

    # Step 3: Eliminate all numbers before object using regexp (e.g '12 0 obj' ==> 'obj')
    new_seq = re.sub(r'\d+ \d+ obj', "obj", new_seq)
    #print(new_seq)

    # Step 4: We also eliminate all \r to reduce size of corpus (optional phase of elimination)
    new_seq = new_seq.replace('\r', '')

    # Return final sequence include all pdf data objects in a single file (pdf_file_path)
    return new_seq


# Main program to call above functions
def main(argv):
    # Counter to keep total number of object that extract from a given directory
    total_extracted_object = 0

    pdf_directory_path = 'D:/iust_pdf_corpus/corpus_garbage/all_00_10kb/'  # 'Dir 1 of IUST corpus' 0-10 kb / == \\ !!!
    pdf_directory_path = 'D:/iust_pdf_corpus/corpus_garbage/drive_deh_10_100kb/'  # 'Dir 2 of IUST corpus' 10-100 kb
    pdf_directory_path = 'D:\\iust_pdf_corpus\\corpus_garbage\\drive_h_100_900kb\\'  # 'Dir 3 of IUST corpus' 100-900 kb
    pdf_directory_path = 'D:\\iust_pdf_corpus\\corpus_garbage\\mozilla\\'  # 'Dir 4 of IUST corpus' mozilla corpus
    object_directory_path = pdf_directory_path + 'pdf_objects\\'

    for filename in os.listdir(pdf_directory_path):
        try:
            # find minimum and maximum object id exist in file filename
            start_index_integer, end_index_integer = get_xref(pdf_directory_path + filename)
            mutool_object_number = ' '
            for obj_id in range(start_index_integer + 1, end_index_integer):
                mutool_object_number += str(obj_id) + ' '
            object_seq = get_pdf_objects(pdf_directory_path + filename, mutool_object_number=mutool_object_number)

            filename_object = object_directory_path + filename + '_' + str(end_index_integer-1) + '_obj.txt'

            with open(filename_object, 'w') as new_file:
                new_file.write(object_seq)

            total_extracted_object += (end_index_integer - 1)
            print('Extracting successfully from %s to %s:' % (filename, filename_object))
        except Exception as e:
            print('Extracting failed from %s:' % filename, file=sys.stderr)
            print(str(e), file=sys.stderr)
            # finally:

    print('total_extracted_object: %s' % total_extracted_object)


if __name__ == "__main__":
    main(sys.argv)

