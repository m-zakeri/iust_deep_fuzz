# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:13:22 2018

@author: Morteza
"""

import sys
import os
import subprocess
import re

pdf_file_name = ''
stream_object_id = 15
cmd = 'D:\\afl\\mupdf-1.11-windows\\mutool.exe show -b -e '
file = 'D:\\afl\\mupdf-1.11-windows\\input\\pdftc_100k_0001.pdf ' + str(stream_object_id) 
cmd += file

returned_value_in_byte = subprocess.check_output(cmd, shell=True)
#return_value_in_string = returned_value_in_byte.decode()

#      can I use regexp?
#     return_value = re.sub(r'stream.*endstream', 'stream', return_value_in_string, flags=re.DOTALL)
#print(return_value_in_string)

#print(returned_value_in_byte)

#x = return_value_in_string.replace('\r','-')
#x = x.replace('\n','=')
#print (x)

stream_file_path = 'D:\\AnacondaProjects\\fuzz_generation\\selected_binary_strams\\'+ 'stream' \
+ str(stream_object_id).zfill(2)
with open(stream_file_path, 'wb') as new_file:
     new_file.write(returned_value_in_byte)

print('stream select successfully', file=sys.stdout)
    
    
    
