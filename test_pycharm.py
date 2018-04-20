import numpy as np
import random

import pdf_object_extractor_2

str1= 'ali'
# print('Hello')

def funv(arg1, arg2):
    print(arg1+arg2)
# funv(2,3)

seq1 = 'x obj yobjzobjcdoj'

# print(seq1.count('obj'))

sample = np.random.multinomial(1, [1/6.]*6)
print('sample 10',sample)

sample = np.random.multinomial(1, [1/5., 1/5., 3/5.], 5)
print('sample 5', sample)
x = random.random()

# print(int(1e6))
# print(int(5.95))

x = 966
x = str(x).zfill(10)
# print(x)

z = 'zekeri morteza'
# print(z.find('mor'))

sum = lambda x,y: x+y
# print(sum(12,13))


# print(pdf_object_extractor_2.get_xref.__doc__)


# Test working with byte and reversing bits of bytes. System programming.
def reverse_all_bits_in_byte():
    x = bytes('xyz120456', encoding='ascii')
    # print('len bytes = ', len(x))
    print('x_all', x)
    print('x5', x[5])
    b = "{0:b}".format(x[5])
    print('x5_binary_str',b)
    breverse =''
    for i in range(len(b)):
        if b[i] == '1':
            breverse+='0'
        else:
            breverse += '1'

    print(breverse)
    breverse2 = int(breverse, 2)
    # breverse = bytes(breverse)
    # breverse2 = bytes([breverse2])
    breverse2 = breverse2.to_bytes(1, 'little')
    print('breverse2',breverse2)
    x = x[:5]+ breverse2 +x[6:]
    print('x5_rev=', x[5])
    print('x_all_new=', x)

