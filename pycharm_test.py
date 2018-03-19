import numpy as np
import random

str1= 'ali'
# print('Hello')

def funv(arg1, arg2):
    print(arg1+arg2)

# funv(2,3)

seq1 = 'x obj yobjzobjcdoj'

# print(seq1.count('obj'))

sample = np.random.multinomial(6, [1/6.]*6, 10)
# print(sample)

sample = np.random.multinomial(1, [1/5., 1/5., 3/5.], 5)
# print(sample)

x = random.random()

# print(int(1e6))

# print(int(5.95))

x = 966

x = str(x).zfill(10)
# print(x)

z = 'zekeri morteza'
print(z.find('mor'))