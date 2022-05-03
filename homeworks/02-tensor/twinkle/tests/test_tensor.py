from math import prod

import numpy as np
from numpy import eye as foo
from numpy.testing import assert_allclose

from ..ops import eye, to_categorical
from ..tensor import Tensor


def test_reshape():
    assert Tensor([0, 1, 2, 3]).reshape(1, 4).shape == (1, 4)
    assert Tensor([0, 1, 2, 3]).reshape(2, 2).shape == (2, 2)
    assert Tensor([0, 1, 2, 3]).reshape(4, 1).shape == (4, 1)

    rng = np.random.default_rng()
    for shape in rng.integers(0, 100, (10, 2)):
        tensor = Tensor([0 for _ in range(prod(shape))])
        reshaped_tensor = tensor.reshape(*shape)
        assert tensor is not reshaped_tensor
        assert_allclose(reshaped_tensor.shape, shape)


def test_flatten():
    t = Tensor([0, 1, 2, 3])
    assert t.reshape(1, 4).flatten().shape == (4,)
    assert t.reshape(2, 2).flatten().shape == (4,)
    assert t.reshape(4, 1).flatten().shape == (4,)

    rng = np.random.default_rng()
    for shape in rng.integers(0, 100, (10, 2)):
        tensor = Tensor([0 for _ in range(prod(shape))]).reshape(*shape)
        flattened_tensor = tensor.flatten()
        assert tensor is not flattened_tensor
        assert_allclose(flattened_tensor.shape, (prod(shape),))


def binary_op_number(fn):
    rng = np.random.default_rng()
    for shape_len in rng.integers(2, 5, (10,)):
        shape = rng.integers(1, 10, (shape_len,))
        data = rng.normal(0, 2, shape).flatten()
        number = rng.normal(0, 2)
        assert_allclose(fn(Tensor(data).reshape(*shape), number),
                        Tensor(fn(data, number) * 1.0).reshape(*shape) * 1.0, rtol=1e-3)
        assert_allclose(fn(number, Tensor(data).reshape(*shape)),
                        Tensor(fn(number, data) * 1.0).reshape(*shape) * 1.0, rtol=1e-3)


def binary_op_scalar_tensor(fn):
    rng = np.random.default_rng()
    for size in rng.integers(2, 5, (10,)):
        shape = rng.integers(1, 10, (size,))
        a, b = rng.normal(0, 2, shape).flatten(), rng.normal(0, 2)
        assert_allclose(fn(Tensor(a).reshape(*shape), Tensor(b)),
                        Tensor(fn(a, b) * 1.0).reshape(*shape) * 1.0, rtol=1e-3)
        assert_allclose(fn(Tensor(b), Tensor(a).reshape(*shape)),
                        Tensor(fn(b, a) * 1.0).reshape(*shape) * 1.0, rtol=1e-3)


def binary_op_tensor(fn):
    rng = np.random.default_rng()
    for size in rng.integers(2, 5, (10,)):
        shape = rng.integers(1, 10, (size,))
        a, b = rng.normal(0, 2, shape).flatten(), rng.normal(0, 2, shape).flatten()
        assert_allclose(fn(Tensor(a).reshape(*shape), Tensor(b).reshape(*shape)),
                        Tensor(fn(a, b) * 1.0).reshape(*shape) * 1.0, rtol=1e-3)
        assert_allclose(fn(Tensor(b).reshape(*shape), Tensor(a).reshape(*shape)),
                        Tensor(fn(b, a) * 1.0).reshape(*shape) * 1.0, rtol=1e-3)


def binary_op(fn):
    binary_op_number(fn)
    binary_op_scalar_tensor(fn)
    binary_op_tensor(fn)


def test_tensor_math():
    binary_op(lambda a, b: a + b)
    binary_op(lambda a, b: a - b)
    binary_op(lambda a, b: a * b)
    binary_op(lambda a, b: a > b)


def test_mm():
    assert Tensor(range(6)).reshape(2, 3) \
               .mm(Tensor(range(6)).reshape(3, 2)) == \
           Tensor([10.0000, 13.0000,
                   28.0000, 40.0000]).reshape(2, 2)
    assert Tensor(range(6)).add(1).reshape(2, 3) \
               .mm(Tensor(range(6)).add(2).reshape(3, 2)) == \
           Tensor([28.0000, 34.0000,
                   64.0000, 79.0000]).reshape(2, 2)

    rng = np.random.default_rng()
    for m in rng.integers(1, 10, (10,)):
        for n, k in zip(rng.integers(1, 10, (10,)), rng.integers(1, 10, (10,))):
            mat1 = rng.normal(0, 10, (n, m))
            mat2 = rng.normal(0, 10, (m, k))
            tensor1 = Tensor(mat1.flatten()).reshape(n, m)
            tensor2 = Tensor(mat2.flatten()).reshape(m, k)
            assert_allclose(tensor1.mm(tensor2), mat1 @ mat2, rtol=1e-3)


def test_magic_mm():
    assert Tensor(range(6)).reshape(2, 3) @ Tensor(range(6)).reshape(3, 2) == \
           Tensor([10.0000, 13.0000,
                   28.0000, 40.0000]).reshape(2, 2)
    assert Tensor(range(6)).reshape(2, 3).add(1) @ Tensor(range(6)).reshape(3, 2).add(2) == \
           Tensor([28.0000, 34.0000,
                   64.0000, 79.0000]).reshape(2, 2)

    rng = np.random.default_rng()
    for m in rng.integers(1, 10, (10,)):
        for n, k in zip(rng.integers(1, 10, (10,)), rng.integers(1, 10, (10,))):
            mat1 = rng.normal(0, 10, (n, m))
            mat2 = rng.normal(0, 10, (m, k))
            tensor1 = Tensor(mat1.flatten()).reshape(n, m)
            tensor2 = Tensor(mat2.flatten()).reshape(m, k)
            assert_allclose(tensor1 @ tensor2, mat1 @ mat2, rtol=1e-3)


def test_getitem():
    data = np.array([0, 1, 2, 3, 4, 5]).reshape(3, 2)
    tensor = Tensor(data.flatten()).reshape(3, 2)

    assert_allclose(tensor[0], data[0])
    assert_allclose(tensor[1], data[1])
    assert_allclose(tensor[2], data[2])

    rng = np.random.default_rng()
    for shape in rng.integers(1, 5, (10, 2)):
        data = rng.normal(0, 10, shape)
        tensor = Tensor(data.flatten()).reshape(*shape)
        for idx in range(shape[0]):
            assert_allclose(tensor[idx], data[idx])


def test_iter():
    data = np.array([0, 1, 2, 3, 4, 5]).reshape(3, 2)
    tensor = Tensor(data.flatten()).reshape(3, 2)

    for item1, item2 in zip(tensor, data):
        assert_allclose(item1[0], item2[0])

    rng = np.random.default_rng()
    for shape in rng.integers(1, 5, (10, 2)):
        data = rng.normal(0, 10, shape)
        tensor = Tensor(data.flatten()).reshape(*shape)
        for item1, item2 in zip(tensor, data):
            assert_allclose(item1[0], item2[0])
        
        
def test_argmax():
    assert Tensor([0]).argmax() == 0
    assert Tensor([0, 4, 2]).argmax() == 1
    assert Tensor([0, 4, 2, 7, 3]).argmax() == 3

    rng = np.random.default_rng()
    for shape in rng.integers(1, 100, (10,)):
        data = rng.normal(0, 10, shape)
        assert Tensor(data).argmax() == data.argmax()


def test_tensor_t():
    assert Tensor([1.0000, 3.0000,
                   2.0000, 4.0000]).reshape(2, 2).T == \
           Tensor([1.0000, 2.0000,
                   3.0000, 4.0000]).reshape(2, 2)
    assert Tensor([1.0000, 4.0000,
                   2.0000, 5.0000,
                   3.0000, 6.0000]).reshape(3, 2).T == \
           Tensor([1.0000, 2.0000, 3.0000,
                   4.0000, 5.0000, 6.0000]).reshape(2, 3)

    rng = np.random.default_rng()
    for shape in rng.integers(1, 10, (10, 2)):
        mat = rng.normal(0, 10, shape)
        assert_allclose(Tensor(mat.flatten()).reshape(*mat.shape).T, mat.T)


def test_len():
    assert len(Tensor(range(6))) == 6
    assert len(Tensor(range(6)).reshape(1, 6)) == 1
    assert len(Tensor(range(6)).reshape(2, 3)) == 2
    assert len(Tensor(range(6)).reshape(3, 2)) == 3
    assert len(Tensor(range(6)).reshape(6, 1)) == 6


def test_eye():
    assert_allclose(eye(1), Tensor([1]).reshape(1, 1))
    assert_allclose(eye(2), Tensor([1, 0,
                                    0, 1]).reshape(2, 2))
    assert_allclose(eye(3), Tensor([1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1]).reshape(3, 3))

    rng = np.random.default_rng()
    for n in rng.integers(1, 100, (10,)):
        assert_allclose(eye(n), np.eye(n))


def test_to_categorical():
    assert_allclose(to_categorical(Tensor([0]), 1),
                    Tensor([1]).reshape(1, 1))
    assert_allclose(to_categorical(Tensor([1, 0]), 2),
                    Tensor([0, 1,
                            1, 0]).reshape(2, 2))
    assert_allclose(to_categorical(Tensor([1, 0, 0, 1]), 2),
                    Tensor([0, 1,
                            1, 0,
                            1, 0,
                            0, 1]).reshape(4, 2))

    rng = np.random.default_rng()
    for num_classes, samples_count in zip(rng.integers(1, 10, (10,)), rng.integers(1, 10, (10,))):
        samples = rng.integers(0, num_classes, (samples_count,))
        assert_allclose(to_categorical(Tensor(samples), num_classes), foo(num_classes)[samples])