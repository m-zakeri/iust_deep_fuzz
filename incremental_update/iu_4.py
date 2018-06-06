"""
Incremental update pdf file
- New in version 4
-- Add online support to generate object from model.
- New in version 3
-- Add binary stream to object which have stream keyword
-- Fuzz binary streams with mutation-based algorithms.
- New in version 2
-- Attach multi-object to end of the pdf file. base on 'portion_of_rewrite_objects' in config.py
"""

__version__ = '0.4.0'
__author__ = 'Morteza'

import sys
import os

import math
import random
import datetime

import PyPDF2

# sys.path.insert(0, '../config.py')
from .. import config
import pdf_object_preprocess as preprocess
from .. import lstm_text_generation_pdf_objs_8


class IncrementalUpdate(object):
    """
    Implement simple way to update a pdf file.
    """
    def __init__(self,
                 host_id=None,
                 object_file_path=config.iu_config['baseline_object_path'],
                 stream_directory_path=config.iu_config['stream_directory_path']):
        """

        :param host_id: Name of host file without postfix, e.g. host1_max, host2_min or host3_avg
        :param object_file_path: See iu_config, new_objects_path
        :param stream_directory_path: See iu_config, stream_directory_path
        """
        self.host_id = host_id

        self.object_file_path = object_file_path
        self.obj_list = preprocess.get_list_of_object(seq=preprocess.load_from_file(self.object_file_path),
                                                      is_sort=False)

        self.stream_directory_path = '../' + stream_directory_path
        self.stream_filename_list = os.listdir(self.stream_directory_path)

        # Creating new directory foreach time that program run and we want to generate new test data
        dt = datetime.datetime.now().strftime(self.host_id + '_date_%Y-%m-%d_%H-%M-%S')
        self.storage_dir_name = config.iu_config['new_pdfs_directory'] + self.host_id + '/' + dt + '/'
        if not os.path.exists(self.storage_dir_name):
            os.makedirs(self.storage_dir_name)
            print('New storage directory build.')

        self.obj_getter = self.obj_generator(self.obj_list)

        retval = os.getcwd()
        os.chdir('../')
        print(os.getcwd())
        self.fff = lstm_text_generation_pdf_objs_8.FileFormatFuzzer(maxlen=50, step=1, batch_size=256)

        self.object_buffer_list = self.fff.load_model_and_generate()
        self.object_buffer_index = 0
        os.chdir(retval)

    def read_pdf_file(self):
        with open(config.iu_config['raw_host_directory'] + self.host_id + '.pdf', 'rb') as f:
            data = f.read()
        return data

    def write_pdf_file(self, name_description, data):
        with open(self.storage_dir_name + self.host_id + name_description + '.pdf', 'wb') as f:
            f.write(data)

    def obj_generator(self, obj_list):
        i = 0
        while True:
            yield obj_list[i]
            i += 1
            if i >= len(obj_list):
                i = 0

    def get_one_object(self, getting_object_policy=config.iu_config['getting_object_policy'], from_model=True):
        """
        Provide one pdf data object whether an existing object in corpus or
        an online new generated object from learnt model
        this function is not complete yet!
        when complete it is expected to get object coming from deep model (learn_and_fuzz) algorithm
        """
        obj = ''
        if from_model:
            obj = self.__get_one_object_from_model()
        else:
            if getting_object_policy == 'sequential':
                # For now using object getter generator
                obj = next(self.obj_getter)
                print(obj)
                # x = input()
            elif getting_object_policy == 'random':
                # For now randomly choose as object from given obj_list
                random_object_index = random.randint(0, len(self.obj_list) - 1)
                obj = self.obj_list[random_object_index]

        # Recently added (1397-01-16)
        # Check if selected object contain keyword 'stream' then add a random/ and fuzzed binary stream to
        # selected object.
        stream_index = obj.find('stream')
        obj = bytes(obj, encoding='ascii')
        if stream_index != -1:
            random_stream_index = random.randint(0, len(self.stream_filename_list)-1)
            with open(self.stream_directory_path+self.stream_filename_list[random_stream_index], mode='rb') as f:
                binary_stream = f.read()
            # Fuzz binary stream separately.
            # We use generation fuzzing for pdf data objects and mutation fuzzing for pdf binary streams that exist
            # within our pdf data objects
            # binary_stream = self.fuzz_binary_stream(binary_stream)
            obj = obj[:stream_index+7] + binary_stream + bytes('\nendstream\n', encoding='ascii') + obj[stream_index+7:]
            print('binary_stream add.')
            # print(obj)

        return obj

    def __get_one_object_from_model(self):
        """

        :return:
        """
        if self.object_buffer_index < len(self.object_buffer_list):
            temp = self.object_buffer_index
            self.object_buffer_index += 1
            return self.object_buffer_list[temp]
        else:
            self.object_buffer_index = 0
            retval = os.getcwd()
            os.chdir('../')
            self.object_buffer_list = self.fff.load_model_and_generate()
            obj = self.__get_one_object_from_model()
            os.chdir(retval)
            return obj

    def get_last_object_id(self):
        with open(config.iu_config['raw_host_directory'] + self.host_id + '.pdf', 'br') as f:
            read_pdf = PyPDF2.PdfFileReader(f)
        last_object_id = read_pdf.trailer['/Size'] - 1  # size xref  - 1
        return last_object_id

    def incremental_update(self, sequential_number=0):
        """
        Shape the incremental update behaviour
        :param sequential_number: Number appear at the end of new pdf file name as identity number (e.g. 1,2 and 3)
        :return: Nothing
        """
        data = self.read_pdf_file()
        last_object_id = str(self.get_last_object_id())
        rewrite_object_content = self.get_one_object()  # Updated. Now include stream objects.

        if config.iu_config['single_object_update']:  # Just one object rewrite with new content
            if config.iu_config['update_policy'] == 'random':
                # Random choose between [2,:] because we don't want modify first object at any condition.
                rewrite_object_id = str(random.randint(2, int(last_object_id)))
            elif config.iu_config['update_policy'] == 'bottom_up':
                rewrite_object_id = last_object_id
            else:
                rewrite_object_id = last_object_id
            data = self.attach_new_object(data=data,
                                          rewrite_object_id=rewrite_object_id,
                                          rewrite_object_content=rewrite_object_content)
            # Set name for new pdf files like:
            # host1_sou_85_6_20180307_114117
            # dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
            name_description = '_sou_' + str(sequential_number).zfill(4) + '_obj-' + str(rewrite_object_id).zfill(3)
            self.write_pdf_file(name_description, data)
            print('save new pdf file successfully')
        else:  # Multiple object rewrite with new content (base on 'portion_of_rewrite_objects') in config file
            number_of_rewrite_objects = math.ceil(config.iu_config['portion_of_rewrite_objects'] * int(last_object_id))
            # print(host_id, number_of_of_rewrite_objects)
            rewrite_object_id = last_object_id
            rewrite_object_ids = ''
            for i in range(int(number_of_rewrite_objects)):
                rewrite_object_content = self.get_one_object()
                if config.iu_config['update_policy'] == 'random':
                    # Random choose between [2,:] because we don't want modify first object at any condition.
                    rewrite_object_id = str(random.randint(2, int(last_object_id)))
                elif config.iu_config['update_policy'] == 'bottom_up':
                    rewrite_object_id = int(last_object_id) - i
                elif config.iu_config['update_policy'] == 'top-down':
                    # Not implement yet.
                    pass
                rewrite_object_ids += '-' + str(rewrite_object_id).zfill(3)
                data = self.attach_new_object(data=data,
                                              rewrite_object_id=rewrite_object_id,
                                              rewrite_object_content=rewrite_object_content)
            # Set name for new pdf files like:
            # host1_sou_85_6_20180307_114117
            # dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
            name_description = '_mou_' + str(sequential_number).zfill(4) + '_objs' + str(rewrite_object_ids)
            self.write_pdf_file(name_description, data)
            print('save new pdf file successfully')

    def attach_new_object(self,
                          data=None,
                          rewrite_object_id=None,
                          rewrite_object_content=None):
        """
        Incremental update pdf file
        :param data:
        :param rewrite_object_id: pdf object id w want to update
        :param rewrite_object_content: new content fo pdf object
        :return:
        """

        # Find last trailer in a pdf file
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

        attach_content = bytes(str(rewrite_object_id + ' 0 '), encoding='ascii')\
                         + rewrite_object_content\
                         + bytes('\nxref\n0 1\n0000000000 65535 f\n'
                                 + rewrite_object_id
                                 + ' 1\n'
                                 + str(len(data)).zfill(10)
                                 + ' 00000 n\n', encoding='ascii')
        attach_content = attach_content \
                         + trailer_content_new \
                         + b'\nstartxref\n'\
                         + bytes(str(startxref_offset_new), encoding='ascii')\
                         + b'\n%%EOF\n'

        # print('attach_content\n', attach_content)
        new_pdf_file = data + attach_content
        return new_pdf_file

    def fuzz_binary_stream(self, binary_stream):
        """
        Fuzzing fuzzing_policy for binary stream fuzz testing.
        Simple basic fuzzing fuzzing_policy:
        Reverse 1% of all bytes in stream randomly. Below code do this
        :param binary_stream:
        :return: fuzzed_binary_stream
        """
        if config.iu_config['stream_fuzzing_policy'] == 'basic_random':
            for i in range(math.ceil(len(binary_stream)/100)):
                # Choose one byte randomly
                byte_to_reverse_index = random.randint(0, len(binary_stream)-1)
                one_byte = binary_stream[byte_to_reverse_index]

                # Convert byte int representation to byte binary string representation (Step 2)
                eight_bit = "{0:b}".format(one_byte)
                # print('eight_bit of one_bye', eight_bit)

                # Reverse eight_bit e.g. 0110000 ==> 1001111
                eight_bit_reverse_str = ''
                for j in range(len(eight_bit)):
                    if eight_bit[j] == '1':
                        eight_bit_reverse_str += '0'
                    else:
                        eight_bit_reverse_str += '1'
                # print('eight_bit_reverse_str', eight_bit_reverse_str)

                # Back eight_bit_reverse_str to int representation (Reverse of step 2)
                eight_bit_reverse_int = int(eight_bit_reverse_str, 2)

                # Convert eight_bit_reverse_int to byte
                one_byte_reverse = \
                    eight_bit_reverse_int.to_bytes(1, 'little')  # 1 is for one byte as length, e.g 15 => 0x0f

                # Substitute one_byte with one_byte_reverse in the  input binary_stream
                binary_stream = binary_stream[0:byte_to_reverse_index]\
                                + one_byte_reverse \
                                + binary_stream[byte_to_reverse_index+1:]
        elif config.iu_config['stream_fuzzing_policy'] == 'other':
            # No other policy implement yet:)
            pass
        return binary_stream


def main(argv):
    host_id = 'host2_min'
    amount_of_testdata = 1000
    iu = IncrementalUpdate(host_id=host_id)
    for i in range(amount_of_testdata):
        iu.incremental_update(sequential_number=i)

    print('%s test data was generate' % amount_of_testdata)


if __name__ == '__main__':
    main(sys.argv)



