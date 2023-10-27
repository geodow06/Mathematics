import unittest
from numpy import exp
from godmathlib.calculus import *

assertions = unittest.TestCase('__init__')


def f1(t, y):
    return 2 * y + 5


A1 = exp(8) - (5 / 2)


def f2(t, y):
    return (-2.2067 * (10 ** -12)) * (y ** 4 - (81 * (10 ** 8)))


A2 = 647.57


class TestIntegration(unittest.TestCase):
    def test_euler_f1(self):
        theta0 = -1.5
        t = 4
        t0 = 0
        steps = 1000000
        assertions.assertAlmostEqual(A1, euler(f1, theta0, t0, t, steps), places=0)

    def test_euler_f2(self):
        theta0 = 1200
        t = 480
        t0 = 0
        steps = 1000000
        assertions.assertAlmostEqual(A2, euler(f2, theta0, t0, t, steps), places=2)

    def test_rk4_f1(self):
        theta0 = -1.5
        t = 4
        t0 = 0
        steps = 1000000
        assertions.assertAlmostEqual(A1, rk4(f1, theta0, t0, t, steps), places=1)

    def test_rk4_f2(self):
        theta0 = 1200
        t = 480
        t0 = 0
        steps = 1000000
        assertions.assertAlmostEqual(A2, rk4(f2, theta0, t0, t, steps), places=2)

    def test_midpoint_f1(self):
        theta0 = -1.5
        t = 4
        t0 = 0
        steps = 1000000
        assertions.assertAlmostEqual(A1, midpoint(f1, theta0, t0, t, steps), places=1)

    def test_midpoint_f2(self):
        theta0 = 1200
        t = 480
        t0 = 0
        steps = 1000000
        assertions.assertAlmostEqual(A2, midpoint(f2, theta0, t0, t, steps), places=2)


if __name__ == '__main__':
    unittest.main()
