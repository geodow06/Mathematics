import numpy as np

__all__ = [
    'transpose', 'multiply', 'dagger', 'determinent',
    'determinent2x2', 'gauss_det', 'list_from_array',
    'gauss_el', 'cross_product', 'trace', 'upper_triangle', 'minor', 'minor_indices',
    'inverse2x2', 'adjugate', 'minor_matrix', 'cofactor', 'inverse'
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


def gauss(m):
    # eliminate columns
    for col in range(len(m[0])):
        for row in range(col + 1, len(m)):
            r = [(rowValue * (-(m[row][col] / m[col][col]))) for rowValue in m[col]]
            m[row] = [sum(pair) for pair in zip(m[row], r)]
    # now backsolve by substitution
    ans = []
    m.reverse()  # makes it easier to backsolve
    for sol in range(len(m)):
        if sol == 0:
            ans.append(m[sol][-1] / m[sol][-2])
        else:
            inner = 0
            # substitute in all known coefficients
            for x in range(sol):
                inner += (ans[x] * m[sol][-2 - x])
            # the equation is now reduced to ax + b = c form
            # solve with (c - b) / a
            ans.append((m[sol][-1] - inner) / m[sol][-sol - 2])
    ans.reverse()
    return ans


def upper_triangle(m):
    # eliminate columns
    for col in range(len(m[0])):
        for row in range(col + 1, len(m)):
            r = [(rowValue * (-(m[row][col] / m[col][col]))) for rowValue in m[col]]
            m[row] = [sum(pair) for pair in zip(m[row], r)]
    # now backsolve by substitution
    return m


def gauss_det(m):
    values = []
    tm = upper_triangle(m)
    for i in range(tm.shape[0]):
        values.append(tm[i, i])
    det = np.prod(values)
    return det


def gauss_el(m):
    ans = []
    upper_triangle(m)
    m.reverse()  # makes it easier to backsolve
    for sol in range(len(m)):
        if sol == 0:
            ans.append(m[sol][-1] / m[sol][-2])
        else:
            inner = 0
            # substitute in all known coefficients
            for x in range(sol):
                inner += (ans[x] * m[sol][-2 - x])
            # the equation is now reduced to ax + b = c form
            # solve with (c - b) / a
            ans.append((m[sol][-1] - inner) / m[sol][-sol - 2])
    ans.reverse()
    return ans
