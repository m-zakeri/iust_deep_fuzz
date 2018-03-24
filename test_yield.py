
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
