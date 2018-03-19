# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:13:22 2018 (1396-10-16)
File name: pdf_object_extractor
@author: Morteza Zakeri

Description:
This file use to extract data object in (set of) PDF files.
We use mutool a subprogram of mupdf an open source pdf library, readers and tools.

"""



import sys
import subprocess
import re


def get_xref(pdf_file_path, mutool_path='D:\\afl\\mupdf-1.11-windows\\mutool.exe',
             mutool_command=' show -e ', mutool_object_number=' x'):
#     command line to get xref size
     cmd = mutool_path + mutool_command + pdf_file_path + mutool_object_number
#     cmd = 'dir'
#     cmd = 'D:\\afl\\mupdf-1.11-windows\\mutool.exe show -e  D:\\afl\\mupdf-1.11-windows\\input\\pdftc_100k_2708.pdf x'

     returned_value_in_byte = subprocess.check_output(cmd, shell=True)
     return_value_in_string = returned_value_in_byte.decode()
#      print command line output to see it in python console
#     print(return_value_in_string)
     start_index_string = ''
     end_index_string = ''
     index = 6
     while(return_value_in_string[index].isdigit()):
          start_index_string += return_value_in_string[index]
          index+=1
     index+=1
     while(return_value_in_string[index].isdigit()):
          end_index_string += return_value_in_string[index]
          index += 1

     start_index_integer = int(start_index_string)
     end_index_integer = int(end_index_string)
#     print(start_index_integer, end_index_integer)
     return start_index_integer,end_index_integer


def get_pdf_objects(pdf_file_path, mutool_path='D:\\afl\\mupdf-1.11-windows\\mutool.exe', 
                        mutool_command=' show -e ', mutool_object_number=' x'):
     cmd = mutool_path + mutool_command + pdf_file_path + mutool_object_number
#    cmd = 'D:\\afl\\mupdf-1.11-windows\\mutool.exe show -e  D:\\afl\\mupdf-1.11-windows\\input\\pdftc_100k_2708.pdf '
#    print(cmd)
     returned_value_in_byte = subprocess.check_output(cmd, shell=True)
     return_value_in_string = returned_value_in_byte.decode()

#      can I use regexp?
#     return_value = re.sub(r'stream.*endstream', 'stream', return_value_in_string, flags=re.DOTALL)
#     print(return_value_in_string)

     stream_start = 0
     stream_end = 0

     new_seq = ''
#     input()
     while(return_value_in_string.find('endstream', stream_end+9) != -1):
          stream_start = return_value_in_string.find('stream', stream_end+9)
          new_seq += return_value_in_string[stream_end:stream_start+6]
          stream_end = return_value_in_string.find('endstream', stream_end+9)
#              if(stream_start != -1 and stream_end != -1):
#              for i in range(stream_start+6,stream_end):
#               return_value_in_list.(i)
#     print(stream_start, stream_end)

     new_seq+= return_value_in_string[stream_end:len(return_value_in_string)]
     new_seq = new_seq.replace('streamendstream', 'stream')
#     print(new_seq)
     new_seq = re.sub(r'\d+ \d+ obj', "obj", new_seq)
#     print(new_seq)
#    need to replace all \r to reduce size of corpus
     new_seq = new_seq.replace('\r','')
     return new_seq


def main(argv):
     total_extracted_object = 0
     pdf_file_path = 'C:\\Users\\Morteza\\Desktop\\corpus_garbage\\mozilla\\' 
     min_pdf_file_id = 1
     max_pdf_file_id = 341
     current_pdf_file_id = min_pdf_file_id
 
     for i in range(min_pdf_file_id, max_pdf_file_id):
          try:
               pdf_file_path += ('pdftc_mozilla_' + str(current_pdf_file_id).zfill(4) +'.pdf')
               start_index_integer,end_index_integer = get_xref(pdf_file_path)
               
               mutool_object_number = ' '
               for index in range (start_index_integer+1, end_index_integer):
                    mutool_object_number += str(index) + ' '
               object_seq = get_pdf_objects(pdf_file_path, mutool_object_number=mutool_object_number)
               
               #print(object_seq)
               #input()
               
               object_file_path = 'C:\\Users\\Morteza\\Desktop\\corpus_garbage\\mozilla' \
               + '\\pdf_objects\\pdftc_mozilla_' \
               + str(current_pdf_file_id).zfill(4) + '_' \
               + str(end_index_integer-1)+ '_obj.pdfobjects'
               #print(object_file_name)

               with open(object_file_path, 'w') as new_file:
                  new_file.write(object_seq)
               total_extracted_object += (end_index_integer - 1)
               print('Extracting successfull: ', str(current_pdf_file_id).zfill(4))
          except Exception as e:
               print('Extracting error: ', str(current_pdf_file_id).zfill(4), file=sys.stderr)
               print(str(e), file=sys.stderr)
          finally:
               current_pdf_file_id += 1
               pdf_file_path = 'C:\\Users\\Morteza\\Desktop\\corpus_garbage\\mozilla\\'
               
     print('total_extracted_object: ', total_extracted_object)


if __name__ == "__main__":
    main(sys.argv)
    
    
    
    
    
    
