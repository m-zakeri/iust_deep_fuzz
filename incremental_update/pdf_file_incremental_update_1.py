"""
incremental update pdf file
https://code.tutsplus.com/tutorials/how-to-work-with-pdf-documents-using-python--cms-25726
"""

import sys
import PyPDF2
from numpy.lib.format import read_array
import pdf_object_preprocess as poc
import random

host_directory = './hosts/'


def get_last_object_id(host_name):
    # pdf_file = open(host_directory + 'host1.pdf', 'br')

    with open(host_directory + host_name + '.pdf', 'br') as f:
        read_pdf = PyPDF2.PdfFileReader(f)

    # print(read_pdf.trailer)
    # print()
    last_object_id = read_pdf.trailer['/Size'] - 1  # size xref  - 1
    # print()
    # print(read_pdf.xref)

    d = read_pdf.xref[0][last_object_id]
    # print(d)
    return last_object_id


def attach_new_object():
    """ incremental update pdf file """
    host_names = ['host1', 'host2', 'host3']
    with open(host_directory + host_names[0] + '.pdf', 'br') as f:
        data = f.read()
        print(len(data))

    # find last trailer in a pdf file
    trailer_index = 0
    while data.find(b'trailer', trailer_index+7) != -1:
        trailer_index = data.find(b'trailer', trailer_index+7)
    print('trailer_index', trailer_index)

    trailer_index_dic_endof = data.find(b'>>', trailer_index)
    print('trailer_index_dic_endof', trailer_index_dic_endof)
    
    trailer_content = data[trailer_index: trailer_index_dic_endof+2]
    print('trailer_content', trailer_content)

    # find last startxref offset in a pdf file
    startxref_index = trailer_index
    while data.find(b'startxref', startxref_index+9) != -1:
        startxref_index = data.find(b'startxref', startxref_index+9)
    # print('index ===', index_startxref)
    index_eof = data.find(b'%%EOF', startxref_index)
    # print('index 2===', index_eof)
    if data[startxref_index+9] == b'\n' or b'\r':
        # print('yes', data[index_startxref+9])
        startxref_index += 10
    if data[index_eof-1] == b'\n' or b'\r':
        index_eof -= 1
    startxref_offset = int(data[startxref_index: index_eof])
    print('startxref_offset', startxref_offset)

    # print(type(trailer_content))
    trailer_content_new = trailer_content[:-2] + b'   /Prev ' \
                          + bytes(str(startxref_offset), 'ascii') + b' \n>>'
    print('trailer_content_new', trailer_content_new)

    # print(bytes(str(startxref_offset), 'ascii'))

    # load the pdf object form file
    seq = poc.load_from_file(host_directory + 'gen_objs_20180221_142612_epochs10_div1.5_step1.txt')
    obj_list = poc.get_list_of_object(seq)
    random_object_index = random.randint(0, len(obj_list)-1)
    obj = obj_list[random_object_index]

    last_object_id = str(get_last_object_id(host_names[0]))

    random_rewrite_object = str(random.randint(1, int(last_object_id)))

    print('len object', len(obj))
    startxref_offset_new = len(data) + 1 + len(random_rewrite_object) + 3 + len(obj) # if we attach just one obj
    print('startxref_offset_new', startxref_offset_new)

    attach_content = bytes(str(random_rewrite_object + ' 0 ' + obj + '\nxref\n0 1\n0000000000 65535 f\n' +\
                               random_rewrite_object + ' 1\n' + str(len(data)).zfill(10) + ' 00000 n\n'), 'ascii') +\
                     trailer_content_new + b'\nstartxref\n' + \
                     bytes(str(startxref_offset_new), 'ascii') + b'\n%%EOF\n'

    print('attach_content\n', attach_content)

    new_pdf_file = data + attach_content
    with open(host_directory + host_names[0] + 'iu_auto7.pdf', 'bw') as f:
        f.write(new_pdf_file)


def main(argv):
    attach_new_object()


if __name__ == '__main__':
    main(sys.argv)
