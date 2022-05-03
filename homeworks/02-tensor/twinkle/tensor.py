from array import array
from collections.abc import Iterable
import numbers
from math import prod

from .linalg import vector_norm


class Tensor(Iterable):
    def __init__(self, initializer):
        if isinstance(initializer, numbers.Number):
            self._data = initializer
            self._shape = ()
        elif isinstance(initializer, Tensor):
            self._data = initializer._data
            self._shape = initializer._shape
        elif isinstance(initializer, Iterable):
            for x in initializer:
                assert isinstance(x, numbers.Number), f"invalid data type '{type(x)}'"
            self._data = array('f', initializer)
            self._shape = (len(self._data),)
        else:
            raise TypeError(f"invalid data type '{type(initializer)}'")

    def __repr__(self):
        if len(self._shape) == 0:
            return f"Tensor({self._data:.4f})"
        elif len(self._shape) == 1:
            return f"Tensor([{', '.join([f'{x:.4f}' for x in self._data])}])"
        elif len(self._shape) == 2:
            rows = [', '.join(f'{x:.4f}' for x in self._data[i * self._shape[1]: (i + 1) * self._shape[1]])\
                    for i in range(self._shape[0])]
            rows = ',\n        '.join(rows)
            return f"Tensor([{rows}]).reshape{self._shape}"
        return f"Tensor([{', '.join([f'{x:.4f}' for x in self._data])}]).reshape{self._shape}"

    def _assert_same_shape(self, other):
        assert isinstance(other, Tensor), f"argument 'other' must be Tensor, not {type(other)}"
        assert self._shape == other._shape, \
            f'The shape of tensor a ({self._shape}) must match the shape of tensor b ({other._shape})'

    def allclose(self, other, rtol=1e-05, atol=1e-08):
        self._assert_same_shape(other)
        return vector_norm(self - other) <= atol + vector_norm(rtol * other)

    # Часть 1:

    @property
    def shape(self):
        pass  # Your code here

    def _binary_op(self, other, fn):
        if isinstance(other, numbers.Number):
            pass  # Your code here
        if isinstance(other, Tensor):
            pass  # Your code here
            self._assert_same_shape(other)
            pass  # Your code here
        raise TypeError(f"unsupported operand type(s) for +: 'Tensor' and '{type(other)}'")

    def add(self, other):
        pass  # Your code here

    def mul(self, other):
        pass  # Your code here

    def sub(self, other):
        pass  # Your code here

    def lt(self, other):
        pass  # Your code here

    def gt(self, other):
        pass  # Your code here

    def neg(self):
        pass  # Your code here

    def dot(self, other):
        self._assert_same_shape(other)
        assert len(self._shape) == 1, '1D tensors expected'
        pass  # Your code here

    def __add__(self, other):
        pass  # Your code here

    def __radd__(self, other):
        pass  # Your code here

    def __mul__(self, other):
        pass  # Your code here

    def __rmul__(self, other):
        pass  # Your code here

    def __sub__(self, other):
        pass  # Your code here

    def __rsub__(self, other):
        pass  # Your code here

    def __gt__(self, other):
        pass  # Your code here

    def __lt__(self, other):
        pass  # Your code here

    def __neg__(self):
        pass  # Your code here

    def __len__(self):
        pass  # Your code here

    def __eq__(self, other):
        pass  # Your code here

    def __iter__(self):
        assert len(self._shape) > 0, 'iteration over a 0-d tensor'
        pass  # Your code here

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            if len(self._shape) == 1:
                pass  # Your code here
            pass  # Your code here
        elif isinstance(key, Tensor) and len(key.shape) == 1:
            pass  # Your code here
        raise TypeError(f'only integers and 1-d Tensors are valid indices (got {type(key)})')

    # Часть 2:

    def reshape(self, *shape):
        assert prod(shape) == len(self._data), \
            f"shape '[{shape}]' is invalid for input of size {len(self)}"
        pass  # Your code here

    def flatten(self):
        pass  # Your code here

    def argmax(self):
        pass  # Your code here

    def mm(self, other):
        assert isinstance(other, Tensor), f"argument 'other' must be Tensor, not {type(other)}"
        assert len(self._shape) == 2, 'self must be a matrix'
        assert len(other._shape) == 2, 'other must be a matrix'
        assert self._shape[1] == other._shape[0], 'self and other shapes cannot be multiplied'
        pass  # Your code here

    def __matmul__(self, other):
        pass  # Your code here

    @property
    def T(self):
        pass  # Your code here
