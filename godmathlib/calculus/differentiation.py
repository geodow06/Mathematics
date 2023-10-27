import numpy as np
from ..constants import *
from math import factorial

__all__ = ['five_point', 'function', 'higher_der', 'bin']


def function(x):
    return x ** 2


def five_point(f, x, h):
    value = (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h) + ((h ** 4) * f(x)) / 30
    return value


def higher_der(f, x, h, n):
    f_n = (1 / h ** n) * sum([(-1) ** (k + n) * bin(n, k) * f(x + k * h) for k in range(n + 1)])
    return f_n


def bin(n, k):
    f = lambda x: factorial(x)
    binomial_coefficient = f(n) / (f(k) * f(n - k))
    return binomial_coefficient
