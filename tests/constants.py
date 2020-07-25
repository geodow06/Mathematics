import numpy as np

__all__ = ['T2X21', 'T2X22', 'T2X2E', 'T3X31', 'T3X32', 'T3X3E', 'T4X41', 'T4X42', 'T4X4E', 'C2X21', 'C2X22']

T2X21 = np.asarray([[5, 8], [3, 8]], dtype=np.complex_)
T2X22 = np.asarray([[3, 8], [8, 9]], dtype=np.complex_)
T2X2E = np.asarray([[79, 112], [73, 96]], dtype=np.complex_)
T3X31 = np.asarray([[10, 20, 10], [4, 5, 6], [2, 3, 5]], dtype=np.complex)
T3X32 = np.asarray([[3, 2, 4], [3, 3, 9], [4, 4, 2]], dtype=np.complex)
T3X3E = np.asarray([[130, 120, 240], [51, 47, 73], [35, 33, 45]], dtype=np.complex)
T4X41 = np.asarray([[5, 7, 9, 10], [2, 3, 3, 8], [8, 10, 2, 3], [3, 3, 4, 8]], dtype=np.complex)
T4X42 = np.asarray([[3, 10, 12, 18], [12, 1, 4, 9], [9, 10, 12, 2], [3, 12, 4, 10]], dtype=np.complex)
T4X4E = np.asarray([[210, 267, 236, 271], [93, 149, 104, 149], [171, 146, 172, 268], [105, 169, 128, 169]],
                   dtype=np.complex)
C2X21 = np.asarray([[2 - 1j, 3], [3 + 2j, 2]], dtype=np.complex_)
C2X22 = np.asarray([[3, 1 + 1j], [8, 4 + 2j]], dtype=np.complex_)
