import numpy as np
import random as rand
from LinearAlgebra.operations import multiply2x2, dagger


def identity(dim):
    size = (dim, dim)
    i = np.zeros(size)
    for j in range(dim):
        i[j, j] = 1
    return i


def issquare(m):
    if m.shape[0] == m.shape[1]:
        return True
    return False


def testArray():
    shape = (2, 2)
    test = np.zeros(shape, dtype=np.complex_)
    for i in range(shape[0]):
        for j in range(shape[1]):
            test[i, j] = complex(rand.randint(0, 5), rand.randint(0, 5))
    return test


def isunitary(m):
    if issquare(m):
        conjugate = dagger(m)
        result = multiply2x2(m, conjugate)
        i = identity(m.shape[0])
        if np.array_equal(result, i):
            return True
    return False


def ishermitian(m):
    if issquare(m):
        size = m.shape[0]
        for i in range(size):
            for j in range(size):
                if m[i, j] != m[j, i].conj():
                    return False
        return True
    return False



