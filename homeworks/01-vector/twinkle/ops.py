from math import prod
from random import gauss

from .tensor import Tensor


def randn(*shape):
    return Tensor([gauss(0, 1) for _ in range(prod(shape))])


# Часть 1:

def ones(*shape):
    pass  # Your code here


def tensor(data):
    pass  # Your code here


def zeros(*shape):
    pass  # Your code here
