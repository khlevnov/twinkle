from array import array
from collections.abc import Iterable, Sized
import numbers


class Tensor(Iterable, Sized):
    def __init__(self, initializer):
        if isinstance(initializer, numbers.Number):
            self._data = array('f', [initializer])
            self._shape = ()
        elif isinstance(initializer, Iterable):
            for x in initializer:
                if not isinstance(x, numbers.Number):
                    raise TypeError(f'an integer is required (got type {type(x)})')
            self._data = array('f', initializer)
            self._shape = (len(self._data),)
        else:
            raise TypeError(f"invalid data type '{type(initializer)}'")

    def __repr__(self):
        if self._shape == ():
            return f'Tensor({self._data:.4f})'
        return f"Tensor([{', '.join([f'{x:.4f}' for x in self._data])}])"

    def _expect_same_shape(self, other):
        if self._shape != other.shape:
            raise RuntimeError(f'The size of tensor a ({self._shape}) \
                must match the size of tensor b ({other.shape})')

    # Part 1:

    @property
    def shape(self):
        return ...  # Your code here

    def __len__(self):
        return ...  # Your code here

    def __getitem__(self, key):
        if isinstance(key, int):
            if len(self._shape) == 1:
                return ...  # Your code here
        raise IndexError(f'only integers and 1-d Tensors are valid indices (got {type(key)})')

    def __iter__(self):
        if self._shape == ():
            raise TypeError('iteration over a 0-d tensor')
        pass  # Your code here

    def item(self):
        if len(self._data) != 1:
            raise ValueError('only one element tensors can be converted to Python scalars')
        return ...  # Your code here

    def _binary_op(self, other, fn):
        if self._shape == ():
            if isinstance(other, numbers.Number):
                return ...  # Your code here
            if isinstance(other, Tensor):
                if other._shape == ():
                    return ...  # Your code here
                return ...  # Your code here
        elif isinstance(other, numbers.Number):
            return ...  # Your code here
        elif isinstance(other, Tensor):
            if other._shape == ():
                return ...  # Your code here
            self._expect_same_shape(other)
            return ...  # Your code here
        raise TypeError(f'unsupported operand type(s)')

    def _unary_op(self, fn):
        if self._shape == ():
            return ...  # Your code here
        return ...  # Your code here

    def add(self, other):
        return ...  # Your code here

    def mul(self, other):
        return ...  # Your code here

    def sub(self, other):
        return ...  # Your code here

    def lt(self, other):
        return ...  # Your code here

    def gt(self, other):
        return ...  # Your code here

    def eq(self, other):
        return ...  # Your code here

    def ne(self, other):
        return ...  # Your code here

    def neg(self):
        return ...  # Your code here

    def __add__(self, other):
        return ...  # Your code here

    def __radd__(self, other):
        return ...  # Your code here

    def __mul__(self, other):
        return ...  # Your code here

    def __rmul__(self, other):
        return ...  # Your code here

    def __sub__(self, other):
        return ...  # Your code here

    def __rsub__(self, other):
        return ...  # Your code here

    def __gt__(self, other):
        return ...  # Your code here

    def __lt__(self, other):
        return ...  # Your code here

    def __eq__(self, other):
        return ...  # Your code here

    def __ne__(self, other):
        return ...  # Your code here

    def __neg__(self):
        return ...  # Your code here

    def dot(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"argument 'other' must be Tensor, not {type(other)}")
        if len(self._shape) != 1 or len(other._shape) != 1:
            raise RuntimeError(f'1D tensors expected, but got {len(self._shape)}D and {len(other._shape)}D tensor')
        self._expect_same_shape(other)
        return ...  # Your code here
