import os

from keras import backend as k
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# b = np.ones((3, 1))
# b[0, 0] = 1
# b[1, 0] = 2
# b[2, 0] = 3
# print(b)
kvar = k.variable(np.array([[1, 2], [3, 4]]), dtype='float32')

y = k.pow(np.e, 4)
x = k.exp(4.)
print(k.eval(y))
print(k.eval(x))

# print(k.eval(x))
# x = k.eval(x)
# x = k.print_tensor(x, 'hi')

# print(x)
# print(type(x))

print(k.floatx())