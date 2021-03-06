import numpy as np

__all__ = [
    'transpose', 'multiply', 'dagger', 'determinent',
    'determinent2x2', 'gauss_det', 'list_from_array',
    'gauss_el', 'cross_product', 'trace', 'upper_triangle', 'minor', 'minor_indices',
    'adjugate', 'minor_matrix', 'cofactor', 'inverse'
]


def multiply(a, b):
    """Multiply two matrices A and B where A is an m x n and B is an n x p returning an m x p"""
    m = a.shape[0]
    p = b.shape[1]
    if a.shape[1] == b.shape[0]:
        n = a.shape[1]
    else:
        n = False
    if n:
        new_shape = (m, p)
        c = np.asarray(
            [sum([a[row, x] * b[x, col] for x in range(p)]) for row in range(m) for col in range(p)]
        )
        return np.reshape(c, new_shape)
    else:
        print("The number of columns of A must equal the number of rows in B")


def transpose(m):
    """Returns the transpose matrix n x m array of a given m x n array"""
    rows = m.shape[0]
    cols = m.shape[1]
    new_shape = (cols, rows)
    t = np.reshape(np.asarray([m[row, col] for col in range(cols) for row in range(rows)]), new_shape)
    return t


def dagger(m):
    """Returns the transpose complex conjugate n x m array of a given m x n array"""
    rows = m.shape[0]
    cols = m.shape[1]
    new_shape = (cols, rows)
    t = np.reshape(np.asarray([m[row, col].conj() for col in range(cols) for row in range(rows)]), new_shape)
    return t


# m is list of matrix entries in
# e.g. ((2,1),(5,2)) is a list [2,1,5,2]
def list_from_array(m):
    values = []
    array = np.asarray([1], dtype=np.complex)
    if type(m) == type(array):

        for row in range(m.shape[0]):
            for column in range(m.shape[1]):
                values.append(m[row, column])
        return values
    else:
        return m


def determinent2x2(m):
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    return det


def determinent(m):
    values = []
    size = m.shape[0]
    if size == 2:
        return determinent2x2(m)
    elif size < 2:
        return 0
    else:
        # Holds for any i where i < size we choose 0
        for j in range(size):
            values.append((-1) ** j * m[0, j] * determinent(minor_matrix(m, 0, j)))
        return sum(values)


def minor(arraylist, size, index):
    indices = minor_indices(size)
    new_minor = []
    for row in range(size - 1):
        for column in range(size - 1):
            new_minor.append(arraylist[((row + 1) * size) + indices[index * (size - 1) + column]])
    return new_minor


def minor_matrix(m, i, j):
    """Returns the (i,j) minor matrix"""
    rows = m.shape[0]
    cols = m.shape[1]
    red_cols = cols - 1
    # Remove jth column
    b = np.delete(m, np.s_[j::cols])
    b = np.reshape(b, (rows, red_cols))
    # # Remove ith row
    c = np.delete(b, np.s_[i * red_cols:(i + 1) * red_cols:])
    c = np.reshape(c, (rows - 1, red_cols))
    return c


# element m[row,column] in list rep is list[row*rows + column]

def minor_indices(size):
    values = []
    for i in range(size):
        columns = []
        for j in range(size - 1):
            columns.append((i + (j + 1)) % size)
        values = values + sorted(columns)
    return values


def cofactor(m):
    """Returns the ij cofactor of matrix"""
    rows = m.shape[0]
    cols = rows
    c = np.asarray([(-1) ** (i + j) * determinent(minor_matrix(m, i, j)) for i in range(rows) for j in range(cols)])
    c = np.reshape(c, m.shape)
    return (c)


def adjugate(m):
    """Adjugate matrix is the Transpose of the ij cofactor of a matrix m"""
    return transpose(cofactor(m))


def inverse(m):
    """Returns the inverse matrix"""
    det = determinent(m)
    if det != 0:
        return 1 / det * adjugate(m)
    else:
        print("This matrix has no inverse as its determinent is zero")


def cross_product(a, b, basis=np.asarray([[1], [1], [1]])):
    """Returns the resultant vector from the cross product of a and b"""
    vectors = [basis, a, b]
    dim = max(a.shape)
    flattener = lambda v: v.flatten()
    m = np.asarray([flattener(vector) for vector in vectors])
    c = np.asarray([m[0, j] * determinent(minor_matrix(m, 0, j)) for j in range(dim)])
    c = np.reshape(c, (dim, 1))
    return c


def trace(m):
    """Returns the Trace of a matrix

    Sums the main diagonal components of a square matrix
    """
    a = m.flat[::m.shape[0] + 1]
    return sum(a)


def upper_triangle(m):
    """Returns the upper triangle form of input matrix"""
    tm = m.copy()
    for col in range(len(tm[0])):
        for row in range(col + 1, len(tm)):
            r = [(rowValue * (-(tm[row][col] / tm[col][col]))) for rowValue in tm[col]]
            tm[row] = [sum(pair) for pair in zip(tm[row], r)]
    return tm


def gauss_det(m):
    """Returns the determinent of an nxn matrix

    For a given matrix A there are three properties namely:
    1. The determinent of an nxn matrix is homogenous of degree n
    2. Interchanging any pair of columns or rows of a matrix multiplies its determinant by −1
    3. Adding a scalar multiple of one column to another column does not change the value of the determinant
    4. If A is a triangular matrix then its determinant equals the product of the diagonal entries

    Which allow us to simplify the computation of det(A) from that of the Laplace expansion which requires
    exponential number of minor determinents to be calculated and can instead transform A into a trigngular
    matrix and calculate the product of the diagonal entries thus calculating the determinent
    """
    tm = upper_triangle(m)
    det = np.prod(tm.flat[::tm.shape[0] + 1])
    return det


def gauss_el(m):
    """Returns a list of solutions for a given set of linear equations in matrix form"""
    ans = []
    t = upper_triangle(m)
    # Reverse makes it easier to backsolve
    t.reverse()
    for sol in range(len(m)):
        if sol == 0:
            ans.append(t[sol][-1] / t[sol][-2])
        else:
            # substitute in all known coefficients x
            inner = sum([ans[x] * t[sol][-2 - x] for x in range(sol)])
            # the equation is now reduced to ax + b = c form
            # solve with (c - b) / a
            ans.append((t[sol][-1] - inner) / t[sol][-sol - 2])
    ans.reverse()
    return ans
