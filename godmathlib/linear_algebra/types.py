import numpy as np
import random as rand
from . import operations
import time

__all__ = ['identity', 'identity_new', 'is_square', 'test_array', 'is_unitary', 'is_hermitian']


def identity(dim):
    size = (dim, dim)
    i = np.zeros(size)
    for j in range(dim):
        i[j, j] = 1
    # i.flat[::dim+1] = 1
    toc = time.perf_counter()
    return i


def identity_new(dim):
    size = (dim, dim)
    i = np.zeros(size)
    i.flat[::dim + 1] = 1
    return i


def is_square(m):
    if m.shape[0] == m.shape[1]:
        return True
    return False


def test_array():
    shape = (2, 2)
    test = np.zeros(shape, dtype=np.complex_)
    for i in range(shape[0]):
        for j in range(shape[1]):
            test[i, j] = complex(rand.randint(0, 5), rand.randint(0, 5))
    return test


def is_unitary(m):
    if is_square(m):
        conjugate = operations.dagger(m)
        result = operations.multiply(m, conjugate)
        i = identity(m.shape[0])
        if np.array_equal(result, i):
            return True
    return False


def is_hermitian(m):
    if is_square(m):
        size = m.shape[0]
        for i in range(size):
            for j in range(size):
                if m[i, j] != m[j, i].conj():
                    return False
        return True
    return False
