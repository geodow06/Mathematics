import unittest, numpy as np
from godmathlib.linear_algebra import operations as op
from ..constants import *


class TestOperations(unittest.TestCase):
    def test_transpose(self):
        test = np.asarray([[1, 2], [3, 4]], dtype=np.complex_)
        expected = np.asarray([[1, 3], [2, 4]], dtype=np.complex_)
        np.testing.assert_equal(op.transpose(test), expected)

    def test_dagger(self):
        test = np.asarray([[1, 2 + 1j], [3, 4 + 2j]], dtype=np.complex_)
        expected = np.asarray([[1, 3], [2 - 1j, 4 - 2j]], dtype=np.complex_)
        np.testing.assert_equal(op.dagger(test), expected)

    def test_multiply2x2(self):
        np.testing.assert_equal(op.multiply(T2X21, T2X22), T2X2E)

    def test_multiply_3x3(self):
        np.testing.assert_equal(op.multiply(T3X31, T3X32), T3X3E, "Incorrect Answer")

    def test_multiply_4x4(self):
        np.testing.assert_equal(op.multiply(T4X41, T4X42), T4X4E, "Incorrect Answer")

    def test_multiply2x2complex(self):
        expected = np.asarray([[30 - 3j, 15 + 7j], [25 + 6j, 9 + 9j]], dtype=np.complex_)
        np.testing.assert_equal(op.multiply(C2X21, C2X22), expected)

    def test_det2x2(self):
        expected = complex(-5, -8)
        np.testing.assert_equal(op.determinent(C2X21), expected)

    def test_det3x3(self):
        test = np.asarray([[2, -3, 1], [2, 0, -1], [1, 4, 5]], dtype=np.complex)
        det = op.determinent(test)
        self.assertEqual(det, 49)

    def test_detnxn4x4(self):
        test = np.asarray([[2, -3, 1, 1], [2, 0, -1, 1], [1, 4, 5, 1], [1, 1, 1, 1]], dtype=np.complex)
        det = op.determinent(test)
        self.assertEqual(det, 18)

    def test_detnxn5x5(self):
        test = np.asarray([[2, -3, 1, 1, 2], [0, -1, 1, 1, 4], [5, 1, 1, 1, 1], [1, 2, 3, 1, 2], [3, 4, 1, 6, 4]],
                          dtype=np.complex)
        det = op.determinent(test)
        self.assertEqual(det, -730)

    def test_detnxn6x6(self):
        test = np.asarray(
            [[2, -3, 1, 1, 2, 3], [0, -1, 1, 1, 4, 1], [5, 1, 1, 1, 1, 2], [1, 2, 3, 1, 2, 5], [3, 4, 1, 6, 4, 7],
             [3, 2, 1, 2, 4, 1]], dtype=np.complex)
        det = op.determinent(test)
        self.assertEqual(det, 196)

    def test_upper_triangle(self):
        test = np.asarray([[8, -6, 2], [-6, 7, -4], [2, -4, 3]], dtype=np.complex)
        expected = np.asarray([[8, -6, 2], [0, 2.5, -2.5], [0, 0, 0]], dtype=np.complex)
        np.testing.assert_equal(op.upper_triangle(test), expected)

    def test_gauss_det(self):
        test = np.asarray([[2, -3, 1, 1, 2], [0, -1, 1, 1, 4], [5, 1, 1, 1, 1], [1, 2, 3, 1, 2], [3, 4, 1, 6, 4]],
                          dtype=np.complex)
        det = op.gauss_det(test)
        self.assertAlmostEqual(det, -730, 3, "Gauss 5x5 determinent")

    def test_gauss_el(self):
        np.testing.assert_equal(op.gauss_el([[2.0, 1.0, -1.0, 8.0],
                                             [-3.0, -1.0, 2.0, -11.0],
                                             [-2.0, 1.0, 2.0, -3.0]]), [2.0, 3.0, -1.0])

    def test_trace(self):
        expected = complex(4, -1)
        np.testing.assert_equal(op.trace(C2X21), expected)

    def test_adjugate(self):
        test = np.asarray([[-3, 2, -5], [-1, 0, -2], [3, -4, 1]])
        expected = np.asarray([[-8, 18, -4], [-5, 12, -1], [4, -6, 2]])
        np.testing.assert_equal(op.adjugate(test), expected)

    def test_minor_matrix1(self):
        test = np.asarray([[-3, 2, -5], [-1, 0, -2], [3, -4, 1]])
        expected = np.asarray([[0, -2], [-4, 1]])
        np.testing.assert_equal(op.minor_matrix(test, 0, 0), expected)

    def test_minor_matrix2(self):
        test = np.asarray([[-3, 2, -5], [-1, 0, -2], [3, -4, 1]])
        expected = np.asarray([[-3, -5], [3, 1]])
        np.testing.assert_equal(op.minor_matrix(test, 1, 1), expected)

    def test_inverse(self):
        test = np.asarray([[3, 0, 2], [2, 0, -2], [0, 1, 1]])
        expected = np.asarray([[0.2, 0.2, 0], [-0.2, 0.3, 1], [0.2, -0.3, 0]])
        np.testing.assert_almost_equal(op.inverse(test), expected)
