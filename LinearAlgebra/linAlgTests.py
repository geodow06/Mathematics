import unittest
from LinearAlgebra.__init__ import *


class TestLinAlg(unittest.TestCase):
    def test_transpose(self):
        test = np.asarray([[1, 2], [3, 4]], dtype=np.complex_)
        expected = np.asarray([[1, 3], [2, 4]], dtype=np.complex_)
        np.testing.assert_equal(transpose(test), expected)

    def test_dagger(self):
        test = np.asarray([[1, 2 + 1j], [3, 4 + 2j]], dtype=np.complex_)
        expected = np.asarray([[1, 3], [2 - 1j, 4 - 2j]], dtype=np.complex_)
        np.testing.assert_equal(dagger(test), expected)

    def test_multiply2x2(self):
        test1 = np.asarray([[5, 8], [3, 8]], dtype=np.complex_)
        test2 = np.asarray([[3, 8], [8, 9]], dtype=np.complex_)
        expected = np.asarray([[79, 112], [73, 96]], dtype=np.complex_)
        np.testing.assert_equal(multiply2x2(test1, test2), expected)

    def test_multiply2x2complex(self):
        test1 = np.asarray([[2 - 1j, 3], [3 + 2j, 2]], dtype=np.complex_)
        test2 = np.asarray([[3, 1 + 1j], [8, 4 + 2j]], dtype=np.complex_)
        expected = np.asarray([[30 - 3j, 15 + 7j], [25 + 6j, 9 + 9j]], dtype=np.complex_)
        np.testing.assert_equal(multiply2x2(test1, test2), expected)

    def test_determinent2x2(self):
        test = np.asarray([[2 - 1j, 3], [3 + 2j, 2]], dtype=np.complex_)
        expected = complex(-5, -8)
        np.testing.assert_equal(determinent2x2(test), expected)

    def test_determinent3x3(self):
        test = np.asarray([[2, -3, 1], [2, 0, -1], [1, 4, 5]], dtype=np.complex)
        det = determinent3x3(test)
        self.assertEqual(det, 49)

    def test_detnxn3x3(self):
        test = [2, -3, 1, 2, 0, -1, 1, 4, 5]
        det = determinent(test)
        self.assertEqual(det, 49)

    def test_detnxn4x4(self):
        test = [2, -3, 1, 1, 2, 0, -1, 1, 1, 4, 5, 1, 1, 1, 1, 1]
        det = determinent(test)
        self.assertEqual(det, 18)

    def test_detnxn5x5(self):
        test = [2, -3, 1, 1, 2, 0, -1, 1, 1, 4, 5, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3, 4, 1, 6, 4]
        det = determinent(test)
        self.assertEqual(det, -730)

    def test_determinent4x4(self):
        test = np.asarray([[2, -3, 1, 1], [2, 0, -1, 1], [1, 4, 5, 1], [1, 1, 1, 1]], dtype=np.complex)
        det = determinent4x4(test)
        self.assertEqual(det, 18)

    # def test_upper_triangle(self):
    #     test = np.asarray([[2, -3, 1], [2, 0, -1], [1, 4, 5]], dtype=np.complex)
    #     expected = np.asarray([[2, -3, 1], [0, 3, -2], [0, 0, 8.16666667]], dtype=np.complex)
    #     np.testing.assert_equal(upper_triangle(test), expected)

    def test_gauss_det(self):
        test = np.asarray([[2, -3, 1, 1, 2], [0, -1, 1, 1, 4], [5, 1, 1, 1, 1], [1, 2, 3, 1, 2], [3, 4, 1, 6, 4]],
                          dtype=np.complex)
        det = gauss_det(test)
        self.assertEqual(det, -730, "Gauss 5x5 determinent")

    def test_gauss_el(self):
        np.testing.assert_equal(gauss_el([[2.0, 1.0, -1.0, 8.0],
                                          [-3.0, -1.0, 2.0, -11.0],
                                          [-2.0, 1.0, 2.0, -3.0]]), [2.0, 3.0, -1.0])

    def test_trace(self):
        test = np.asarray([[2 - 1j, 3], [3 + 2j, 2]], dtype=np.complex_)
        expected = complex(4, -1)
        np.testing.assert_equal(trace(test), expected)

    def test_true_isunitary(self):
        test = np.asarray([[0, 1j], [1j, 0]], dtype=np.complex_)
        self.assertTrue(isunitary(test))

    def test_false_isunitary(self):
        test = np.asarray([[2 - 1j, 3], [3 + 2j, 2]], dtype=np.complex_)
        self.assertFalse(isunitary(test))

    def test_true_ishermitian1(self):
        test = np.asarray([[1, -1j], [1j, 1]], dtype=np.complex_)
        self.assertTrue(ishermitian(test))

    def test_true_ishermitian2(self):
        test = np.asarray([[-1, 1 - 2j, 0], [1 + 2j, 0, -1j], [0, 1j, 1]], dtype=np.complex_)
        self.assertTrue(ishermitian(test))

    def test_false_ishermitian1(self):
        test = np.asarray([[2 - 1j, 3], [3 + 2j, 2]], dtype=np.complex_)
        self.assertFalse(ishermitian(test))

    def test_false_ishermitian2(self):
        test = np.asarray([[2 - 1j, 3, 1], [1, -3j, 1 - 3j], [3, 1 + 2j, 2]], dtype=np.complex_)
        self.assertFalse(ishermitian(test))

    # def test_inverse(self):
    #     test = np.asarray([[2 - 1j, 3], [3 + 2j, 2]], dtype=np.complex_)
    #     expected = np.asarray([[2 - 1j, 3], [3 + 2j, 1 + 2j]], dtype=np.complex_)
    #     expected_factor = complex(8 / 89, 5 / 89)
    #     factor, matrix = inverse2x2(test)
    #     np.testing.assert_equal(matrix, expected)
    #     np.testing.assert_equal(inverse2x2(test), expected)


if __name__ == '__main__':
    unittest.main()
