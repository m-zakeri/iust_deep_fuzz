# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 00:32:17 2017

@author: Morteza
"""

import base64

## part 1 convert binary to text using base64 encoding

#read binary file
with open("png1.png", "rb") as pdf_file:
    ascii_byte_string = base64.b64encode(pdf_file.read())

#print (ascii_byte_string)

with open("base64text.txt", encoding='utf-8', mode='w') as text_file:
    t = str(ascii_byte_string,'utf-8')
    text_file.write(t)

print("part1 done!")


## part 2 convert text to binary using base64 decoding

# read text file
with open("base64text.txt", encoding='utf-8', mode='r') as text_file:
    x = text_file.read()
    
x = base64.b64decode(x)
#print(x)
with open("original_binary_file.png", "wb") as orginal_binary_file:
    orginal_binary_file.write(x)
    
print("part2 done!")
