import os, datetime

test_list = [12,13,14,15,16,17,18,19,20]


def fib(a, b):
    # a, b = 0, 1
    j = 1
    while 1:
        yield b
        a, b = b, a+b
        j += 1
        print('j',j)


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

dt = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
dir_name = './generated_results/pdfs/' + dt + '/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

print('build dirs')


def buildCodeBook():
    letters ='.abcdefghijklnopqrstuvwxyz'
    codeBook = {}
    key = 0
    for c in letters:
        codeBook[key] = c
        key += 1
    return codeBook


def decode(cypherText, codeBook):
    plainText = ''
    for e in cypherText:
        if e in codeBook:
            plainText += codeBook[e]
        else:
            plainText += ' '
    return plainText


codeBook = buildCodeBook()
msg = (3,2,41,1,0)
print(decode(msg, codeBook))
