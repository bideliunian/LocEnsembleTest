from os import getpid

def double(i, j):
    print("I'm process", getpid())
    return i * 2 + j
