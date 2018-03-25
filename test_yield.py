
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