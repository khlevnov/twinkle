from math import prod
from random import gauss

from .tensor import Tensor


def randn(*shape):
    return Tensor([gauss(0, 1) for _ in range(prod(shape))]).reshape(*shape)


# Часть 1:

def ones(*shape):
    pass  # Your code here


def tensor(data):
    pass  # Your code here


def zeros(*shape):
    pass  # Your code here


# Часть 2:

def eye(n):
    pass  # Your code here


def to_categorical(y, num_classes):
    pass  # Your code here
