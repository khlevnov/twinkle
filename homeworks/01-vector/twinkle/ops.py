from math import prod
from random import gauss

from .tensor import Tensor


def randn(*shape):
    return Tensor([gauss(0, 1) for _ in range(prod(shape))])


# Part 1:

def ones(*shape):
    return ...  # Your code here


def tensor(data):
    return ...  # Your code here


def zeros(*shape):
    return ...  # Your code here
