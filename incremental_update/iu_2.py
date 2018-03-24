"""
incremental update pdf file
version 2 - attach multi-object to end of the pdf file
base on 'portion_of_rewrite_objects' in config.py
"""


from incremental_update.config import config

import sys
import PyPDF2
from numpy.lib.format import read_array
import pdf_object_preprocess as poc
import random
import datetime
import math
import re


def read_pdf_file(host_id):
    with open(config['raw_host_directory'] + host_id + '.pdf', 'br') as f:
        data = f.read()
    return data


def write_pdf_file(host_id, description ,new_pdf_file):
    with open(config['new_host_directory'] + host_id + '/'
              + host_id + description + '.pdf', 'bw') as f:
        f.write(new_pdf_file)


def get_last_object_id(host_id):
    with open(config['raw_host_directory'] + host_id + '.pdf', 'br') as f:
        read_pdf = PyPDF2.PdfFileReader(f)
    last_object_id = read_pdf.trailer['/Size'] - 1  # size xref  - 1
    return last_object_id


def get_one_object():
    """ provide one pdf data object whether an existing object in corpus or
    an online new generated object from learnt model
    this function is not complete yet!
    """
    object_file_path = '../trainset/pdf_object_trainset_100_to_500_percent33.txt'

    seq = poc.load_from_file(object_file_path)
    obj_list = poc.get_list_of_object(seq, is_sort=False)
    random_object_index = random.randint(50, len(obj_list) - 1)
    obj = obj_list[random_object_index]
    return obj


def incremental_update(single_object_update, host_id, sequential_number):
    """ shape the incremental update """
    data = read_pdf_file(host_id)
    last_object_id = str(get_last_object_id(host_id))
    rewrite_object_content = get_one_object()

    if single_object_update:
        if config['update_policy'] == 'random':
            rewrite_object_id = str(random.randint(1, int(last_object_id)))
        elif config['update_policy' == 'bottom_up']:
            rewrite_object_id = last_object_id
        data = attach_new_object(data, last_object_id,
                                             rewrite_object_content, rewrite_object_id)
        # set name for new pdf files like:
        # host1_sou_85_6_20180307_114117
        dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
        name_description = '_sou_' + str(sequential_number) + '_' + str(rewrite_object_id) + dt
        write_pdf_file(host_id, name_description, data)
        print('save new pdf file successfully')
    else:
        number_of_of_rewrite_objects = math.ceil(config['portion_of_rewrite_objects'] * int(last_object_id))
        # print(host_id, number_of_of_rewrite_objects)
        rewrite_object_ids = ''
        for i in range(number_of_of_rewrite_objects):
            rewrite_object_content = get_one_object()
            if config['update_policy'] == 'random':
                rewrite_object_id = str(random.randint(1, int(last_object_id)))
            elif config['update_policy' == 'bottom_up']:
                rewrite_object_id = last_object_id - i
            rewrite_object_ids += rewrite_object_id
            data = attach_new_object(data, last_object_id,
                                             rewrite_object_content, rewrite_object_id)
            # set name for new pdf files like:
            # host1_sou_85_6_20180307_114117
        dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
        name_description = '_mou_' + str(sequential_number) + '_' + str(rewrite_object_ids) + dt
        write_pdf_file(host_id, name_description, data)
        print('save new pdf file successfully')


def attach_new_object(data, last_object_id, rewrite_object_content, rewrite_object_id):
    """ incremental update pdf file """

    # find last trailer in a pdf file
    trailer_index = 0
    while data.find(b'trailer', trailer_index + 7) != -1:
        trailer_index = data.find(b'trailer', trailer_index + 7)
    print('trailer_index', trailer_index)

    trailer_index_dic_endof = data.find(b'>>', trailer_index)
    print('trailer_index_dic_endof', trailer_index_dic_endof)

    trailer_content = data[trailer_index: trailer_index_dic_endof + 2]
    print('trailer_content', trailer_content)

    # find last startxref offset in a pdf file
    startxref_index = trailer_index
    while data.find(b'startxref', startxref_index + 9) != -1:
        startxref_index = data.find(b'startxref', startxref_index + 9)
    # print('index ===', index_startxref)
    index_eof = data.find(b'%%EOF', startxref_index)
    # print('index 2===', index_eof)
    if data[startxref_index + 9] == b'\n' or b'\r':
        # print('yes', data[index_startxref+9])
        startxref_index += 10
    if data[index_eof - 1] == b'\n' or b'\r':
        index_eof -= 1
    startxref_offset = int(data[startxref_index: index_eof])
    print('startxref_offset', startxref_offset)

    # print(type(trailer_content))
    # remove all /Prev 1234 from trailer if exist
    # trailer_content = trailer_content.replace(b'/Prev', b'')
    # trailer_content = re.sub(r'/Prev \d+', b'', str(trailer_content))
    index_prev = trailer_content.find(b'/Prev')
    if index_prev != -1:
        index_curr = 0
        # print('##', trailer_content[index_prev+5], index_prev)
        # check whether a byte is ascii number or space
        eliminate_content = trailer_content[index_prev+5+index_curr]
        print('eliminate content', eliminate_content)
        while (48 <= eliminate_content <= 57) or (eliminate_content == 32):
            print('###', trailer_content[index_prev+5+index_curr])
            index_curr += 1
            eliminate_content = trailer_content[index_prev + 5 + index_curr]

        trailer_content = trailer_content[:index_prev] + trailer_content[index_prev+5+index_curr:]

    trailer_content_new = trailer_content[:-2] + b'   /Prev ' \
                          + bytes(str(startxref_offset), 'ascii') + b' \n>>'
    print('trailer_content_new', trailer_content_new)

    print('len_rewrite_object_content', len(rewrite_object_content))
    startxref_offset_new = len(data) + 1 + len(rewrite_object_id) + 3 + len(rewrite_object_content)  # if we attach just one obj
    print('startxref_offset_new', startxref_offset_new)

    attach_content = bytes(str(rewrite_object_id + ' 0 ' + rewrite_object_content + '\nxref\n0 1\n0000000000 65535 f\n' + \
                               rewrite_object_id + ' 1\n' + str(len(data)).zfill(10) + ' 00000 n\n'), 'ascii') + \
                     trailer_content_new + b'\nstartxref\n' + \
                     bytes(str(startxref_offset_new), 'ascii') + b'\n%%EOF\n'

    # print('attach_content\n', attach_content)
    new_pdf_file = data + attach_content
    return new_pdf_file


def main(argv):
    host_id = 'host2'
    for i in range(0, 10):
        incremental_update(config['single_object_update'], host_id, i)

    print('*** end ***')


if __name__ == '__main__':
    main(sys.argv)



