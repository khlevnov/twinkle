from array import array
from collections.abc import Iterable
import numbers

from .linalg import vector_norm


class Tensor(Iterable):
    def __init__(self, initializer):
        if isinstance(initializer, numbers.Number):
            self._data = initializer
            self._shape = ()
        elif isinstance(initializer, Iterable):
            for x in initializer:
                assert isinstance(x, numbers.Number), f"invalid data type '{type(x)}'"
            self._data = array('f', initializer)
            self._shape = (len(self._data),)
        else:
            raise TypeError(f"invalid data type '{type(initializer)}'")

    def __repr__(self):
        if len(self._shape) == 0:
            return f'Tensor({self._data:.4f})'
        return f"Tensor([{', '.join([f'{x:.4f}' for x in self._data])}])"

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
            pass  # Your code here
        raise TypeError(f'only integers and 1-d Tensors are valid indices (got {type(key)})')
