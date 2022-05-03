import numpy as np
from numpy.testing import assert_allclose

from ..linalg import vector_norm
from ..ops import zeros, ones, tensor
from ..tensor import Tensor


def test_shape():
    assert Tensor([0]).shape == (1,)
    assert Tensor([0, 1]).shape == (2,)
    assert Tensor([0, 1, 2]).shape == (3,)
    assert Tensor([0, 1, 2, 3]).shape == (4,)

    rng = np.random.default_rng()
    for len_ in rng.integers(0, 100, (10,)):
        assert Tensor([0] * len_).shape == (len_,)


def test_len():
    assert len(Tensor([0])) == 1
    assert len(Tensor([0, 1])) == 2
    assert len(Tensor([0, 1, 2])) == 3
    assert len(Tensor([0, 1, 2, 3])) == 4

    rng = np.random.default_rng()
    for len_ in rng.integers(0, 100, (10,)):
        assert len(Tensor([0] * len_)) == len_


def test_eq():
    assert Tensor([0]) == Tensor([0])
    assert Tensor([0, 1]) == Tensor([0, 1])
    assert Tensor([0, 1, 2]) == Tensor([0, 1, 2])
    assert not Tensor([0, 1, 2]) == Tensor([1, 2, 3])
    assert not Tensor([0, 1, 2]) == Tensor([2, 3, 4])

    rng = np.random.default_rng()
    for vec1, vec2 in zip(rng.normal(0, 100, (10, 10)), rng.normal(0, 10, (10, 10))):
        assert Tensor(vec1) == Tensor(vec1)
        assert Tensor(vec2) == Tensor(vec2)
        assert not Tensor(vec1) == Tensor(vec2)


def test_getitem():
    assert Tensor([3, 2, 1])[0] == 3
    assert Tensor([3, 2, 1])[1] == 2
    assert Tensor([3, 2, 1])[2] == 1

    rng = np.random.default_rng()
    for vec, idx in zip(rng.integers(0, 100, (10, 10)), rng.integers(0, 10, (10,))):
        assert Tensor(vec)[int(idx)] == vec[idx]


def test_iter():
    for a, b in zip(Tensor([0, 1, 2]), [0, 1, 2]):
        assert a == b

    rng = np.random.default_rng()
    for data in rng.integers(0, 100, (10, 10)):
        for a, b in zip(Tensor(data), data):
            assert a == b


def test_vector_norm_l1():
    assert_allclose(vector_norm(Tensor([0]), 1), 0, rtol=1e-5)
    assert_allclose(vector_norm(Tensor([0, 1]), 1), 1, rtol=1e-5)
    assert_allclose(vector_norm(Tensor([0, 1, 2]), 1), 3, rtol=1e-5)
    assert_allclose(vector_norm(Tensor([0, 1, 2, 3]), 1), 6, rtol=1e-5)

    rng = np.random.default_rng()
    for data in rng.normal(0, 10, (10, 10)):
        assert_allclose(vector_norm(Tensor(data), 1), np.linalg.norm(np.array(data), 1))


def test_vector_norm_l2():
    assert_allclose(vector_norm(Tensor([0]), 2), 0, rtol=1e-5)
    assert_allclose(vector_norm(Tensor([0, 1]), 2), 1, rtol=1e-5)
    assert_allclose(vector_norm(Tensor([0, 1, 2]), 2), 2.236068, rtol=1e-5)
    assert_allclose(vector_norm(Tensor([0, 1, 2, 3]), 2), 3.741657, rtol=1e-5)

    rng = np.random.default_rng()
    for data in rng.normal(0, 10, (10, 10)):
        assert_allclose(vector_norm(Tensor(data), 2), np.linalg.norm(np.array(data), 2))


def test_p_norm():
    assert_allclose(vector_norm(Tensor([0]), 1), 0, rtol=1e-5)
    assert_allclose(vector_norm(Tensor([0, 1]), 2), 1, rtol=1e-5)
    assert_allclose(vector_norm(Tensor([0, 1, 2]), 3), 2.080084, rtol=1e-5)
    assert_allclose(vector_norm(Tensor([0, 1, 2, 3]), 4), 3.146346, rtol=1e-5)

    rng = np.random.default_rng()
    for data, p in zip(rng.normal(0, 10, (10, 10)), rng.integers(1, 10, (10,))):
        assert_allclose(vector_norm(Tensor(data), p), np.linalg.norm(np.array(data), p))


def test_zeros():
    assert_allclose(zeros(0), [])
    assert_allclose(zeros(1), [0])
    assert_allclose(zeros(2), [0, 0])
    assert_allclose(zeros(3), [0, 0, 0])

    rng = np.random.default_rng()
    for zeros_count in rng.integers(0, 100, (10,)):
        assert_allclose(zeros(zeros_count), [0 for _ in range(zeros_count)])


def test_ones():
    assert_allclose(ones(0), [])
    assert_allclose(ones(1), [1])
    assert_allclose(ones(2), [1, 1])
    assert_allclose(ones(3), [1, 1, 1])

    rng = np.random.default_rng()
    for zeros_count in rng.integers(0, 100, (10,)):
        assert_allclose(ones(zeros_count), [1 for _ in range(zeros_count)])


def test_tensor():
    assert_allclose(tensor([0]), [0])
    assert_allclose(tensor([0, 1]), [0, 1])
    assert_allclose(tensor([0, 1, 2]), [0, 1, 2])
    assert_allclose(tensor([0, 1, 2, 3]), [0, 1, 2, 3])

    rng = np.random.default_rng()
    for data in rng.normal(0, 10, (10, 100)):
        assert_allclose(tensor(data), data)


def binary_op_number(fn):
    rng = np.random.default_rng()
    for vec, number in zip(rng.normal(0, 100, (10, 10)), rng.normal(0, 10, (10,))):
        assert_allclose(fn(Tensor(vec), number), Tensor(fn(vec, number) * 1.0), rtol=1e-4)
        assert_allclose(fn(float(number), Tensor(vec)), Tensor(fn(number, vec) * 1.0), rtol=1e-4)


def binary_op_scalar_tensor(fn):
    rng = np.random.default_rng()
    for a, b in zip(rng.normal(0, 100, (10, 10)), rng.normal(0, 10, (10,))):
        assert_allclose(fn(Tensor(a), Tensor(b)), Tensor(fn(a, b) * 1.0), rtol=1e-4)
        assert_allclose(fn(Tensor(b), Tensor(a)), Tensor(fn(b, a) * 1.0), rtol=1e-4)


def binary_op_tensor(fn):
    rng = np.random.default_rng()
    for a, b in zip(rng.normal(0, 100, (10, 10)), rng.normal(0, 10, (10, 10))):
        assert_allclose(fn(Tensor(a), Tensor(b)), Tensor(fn(a, b) * 1.0), rtol=1e-4)
        assert_allclose(fn(Tensor(b), Tensor(a)), Tensor(fn(b, a) * 1.0), rtol=1e-4)


def binary_op(fn):
    binary_op_number(fn)
    binary_op_scalar_tensor(fn)
    binary_op_tensor(fn)


def test_add_number():
    assert_allclose(Tensor([0]) + 1, Tensor([1]), rtol=1e-4)
    assert_allclose(Tensor([0, 1]) + 2, Tensor([2, 3]), rtol=1e-4)
    assert_allclose(Tensor([0, 1, 2]) + 3, Tensor([3, 4, 5]), rtol=1e-4)
    binary_op_number(lambda a, b: a + b)


def test_add_scalar_tensor():
    assert_allclose(Tensor([0]) + Tensor(1), Tensor([1]), rtol=1e-4)
    assert_allclose(Tensor([0, 1]) + Tensor(2), Tensor([2, 3]), rtol=1e-4)
    assert_allclose(Tensor([0, 1, 2]) + Tensor(3), Tensor([3, 4, 5]), rtol=1e-4)
    binary_op_scalar_tensor(lambda a, b: a + b)


def test_add_tensor():
    assert_allclose(Tensor([0]) + Tensor([1]), Tensor([1]), rtol=1e-4)
    assert_allclose(Tensor([0, 1]) + Tensor([2, 3]), Tensor([2, 4]), rtol=1e-4)
    assert_allclose(Tensor([0, 1, 2]) + Tensor([3, 4, 5]), Tensor([3, 5, 7]), rtol=1e-4)
    binary_op_scalar_tensor(lambda a, b: a + b)


def test_add():
    binary_op(lambda a, b: a + b)


def test_sub():
    binary_op(lambda a, b: a - b)


def test_mul():
    binary_op(lambda a, b: a * b)


def test_gt():
    binary_op(lambda a, b: a > b)


def test_neg():
    assert_allclose(-Tensor([0]), Tensor([0]), rtol=1e-4)
    assert_allclose(-Tensor([0, 1]), Tensor([0, -1]), rtol=1e-4)
    assert_allclose(-Tensor([0, 1, 2]), Tensor([0, -1, -2]), rtol=1e-4)

    rng = np.random.default_rng()
    for vec in rng.normal(0, 100, (10, 10)):
        assert_allclose(-Tensor(vec), Tensor(-vec), rtol=1e-4)


def test_dot():
    assert_allclose(Tensor([0]).dot(Tensor([0])), 0.0)
    assert_allclose(Tensor([0, 1]).dot(Tensor([0, 1])), 1.0)
    assert_allclose(Tensor([0, 1, 2]).dot(Tensor([0, 1, 2])), 5.0)

    rng = np.random.default_rng()
    for vec1, vec2 in zip(rng.normal(0, 100, (10, 10)), rng.normal(0, 10, (10, 10))):
        assert_allclose(Tensor(vec1).dot(Tensor(vec2)), vec1.dot(vec2), rtol=1e-4)
        assert_allclose(Tensor(vec2).dot(Tensor(vec1)), vec2.dot(vec1), rtol=1e-4)
