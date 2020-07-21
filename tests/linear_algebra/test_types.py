import unittest, numpy as np
from godmathlib.linear_algebra import types as tps

class TestTypes(unittest.TestCase):
    
    def test_true_isunitary(self):
        test = np.asarray([[0, 1j], [1j, 0]], dtype=np.complex_)
        self.assertTrue(tps.is_unitary(test))

    def test_false_isunitary(self):
        test = np.asarray([[2 - 1j, 3], [3 + 2j, 2]], dtype=np.complex_)
        self.assertFalse(tps.is_unitary(test))

    def test_true_ishermitian1(self):
        test = np.asarray([[1, -1j], [1j, 1]], dtype=np.complex_)
        self.assertTrue(tps.is_hermitian(test))

    def test_true_ishermitian2(self):
        test = np.asarray([[-1, 1 - 2j, 0], [1 + 2j, 0, -1j], [0, 1j, 1]], dtype=np.complex_)
        self.assertTrue(tps.is_hermitian(test))

    def test_false_ishermitian1(self):
        test = np.asarray([[2 - 1j, 3], [3 + 2j, 2]], dtype=np.complex_)
        self.assertFalse(tps.is_hermitian(test))

    def test_false_ishermitian2(self):
        test = np.asarray([[2 - 1j, 3, 1], [1, -3j, 1 - 3j], [3, 1 + 2j, 2]], dtype=np.complex_)
        self.assertFalse(tps.is_hermitian(test))

    # def test_inverse(self):
    #     test = np.asarray([[2 - 1j, 3], [3 + 2j, 2]], dtype=np.complex_)
    #     expected = np.asarray([[2 - 1j, 3], [3 + 2j, 1 + 2j]], dtype=np.complex_)
    #     expected_factor = complex(8 / 89, 5 / 89)
    #     factor, matrix = inverse2x2(test)
    #     np.testing.assert_equal(matrix, expected)
    #     np.testing.assert_equal(inverse2x2(test), expected)


if __name__ == '__main__':
    unittest.main()