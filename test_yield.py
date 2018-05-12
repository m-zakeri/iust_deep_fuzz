import os
import datetime

import numpy as np
import random

test_list = [12, 13, 14, 15, 16, 17, 18, 19, 20]


def fib(a, b):
    # a, b = 0, 1
    j = 1
    while 1:
        yield b
        a, b = b, a+b
        j += 1
        print('j', j)


x = fib(0,1)
for i in range(10):
    print(next(x))


# A comment for test ...
def sample_function():
    """ The sample function doc string
      # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    """
    pass


print(sample_function.__doc__)

def test():
    dt = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    dir_name = './generated_results/pdfs/' + dt + '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print('build dirs')


def data_generator(self, sentences, next_chars):
    """
    Batch data generator for large dataset not fit completely in memory
    # Index j now increase sequentially

    :param sentences:
    :param next_chars:
    :return:
    """
    # j = 0
    # print('Vectorization...')
    while True:
        # Fix generator :))
        x = np.zeros((self.batch_size, self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((self.batch_size, len(self.chars)), dtype=np.bool)
        j = random.randint(0, len(sentences) - (self.batch_size + 1))
        for i, one_sample in enumerate(sentences[j: j + self.batch_size]):
            for t, char in enumerate(one_sample):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1
        yield x, y
        # j += self.batch_size
        # if j > (len(sentences) - (self.batch_size+1)):
        #     j = random.randint(0, len(sentences) - (self.batch_size+1))
