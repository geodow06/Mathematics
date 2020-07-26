import numpy as np

__all__ = [
    'transpose', 'multiply', 'dagger', 'determinent', 'determinent2x2list',
    'determinent2x2', 'determinent3x3', 'determinent4x4', 'gauss_det', 'list_from_array',
    'gauss_el', 'cross_product', 'trace', 'upper_triangle', 'minor', 'minor_indices',
    'inverse2x2', 'adjugate','minor_matrix','cofactor','inverse'
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


def determinent2x2(m):
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    return det


def determinent2x2list(l):
    det = l[0] * l[3] - l[1] * l[2]
    return det


def determinent3x3(m):
    values = []
    indices = minor_indices(m.shape[0])
    for i in range(m.shape[0]):
        minor = np.asarray(
            [[m[1, indices[2 * i]], m[1, indices[2 * i + 1]]], [m[2, indices[2 * i]], m[2, indices[2 * i + 1]]]],
            dtype=np.complex)
        values.append((-1) ** i * m[0, i] * determinent2x2(minor))
    return sum(values)


def determinent4x4(m):
    values = []
    size = m.shape[0]
    indices = minor_indices(size)
    for i in range(size):
        minor = np.asarray(
            [[m[1, indices[(size - 1) * i]], m[1, indices[(size - 1) * i + 1]], m[1, indices[(size - 1) * i + 2]]],
             [m[2, indices[(size - 1) * i]], m[2, indices[(size - 1) * i + 1]], m[2, indices[(size - 1) * i + 2]]],
             [m[3, indices[(size - 1) * i]], m[3, indices[(size - 1) * i + 1]], m[3, indices[(size - 1) * i + 2]]]],
            dtype=np.complex)
        values.append((-1) ** i * m[0, i] * determinent3x3(minor))
    return sum(values)


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


def determinent(m):
    values = []
    list = list_from_array(m)
    size = int(len(list) ** 0.5)
    if size == 2:
        return determinent2x2list(list)
        # return determinent2x2(m)
    elif size < 2:
        return 0
    else:
        for i in range(size):
            values.append((-1) ** i * list[i] * determinent(minor(list, size, i)))
            # values.append((-1) ** i * list[i] * determinent(minor_matrix(m, i, j)))
        return sum(values)


def minor(arraylist, size, index):
    indices = minor_indices(size)
    new_minor = []
    for row in range(size - 1):
        for column in range(size - 1):
            new_minor.append(arraylist[((row + 1) * size) + indices[index * (size - 1) + column]])
    return new_minor

def minor_matrix(m,i,j):
    """Returns the (i,j) minor matrix"""
    rows = m.shape[0]
    cols = m.shape[1]
    red_cols = cols-1
    # Remove jth column
    b = np.delete(m, np.s_[j::cols])
    b = np.reshape(b, (rows, red_cols))
    # # Remove ith row
    c = np.delete(b, np.s_[i*red_cols:(i+1)*red_cols:])
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
    # c = np.zeros(m.shape, dtype=np.complex_)
    rows = m.shape[0]
    cols = rows
    list = list_from_array(m)
    size = int(len(list) ** 0.5)
    c = [(-1) ** (i + j) * determinent(minor_matrix(m, i, j)) for i in range(rows) for j in range(cols)]
    c = np.asarray(c)
    c = np.reshape(c, m.shape)
    return (c)

def adjugate(m):
    """Adjugate matrix is the Transpose of the ij cofactor of a matrix m"""
    return transpose(cofactor(m))

def inverse(m):
    """Returns the inverse matrix"""
    det = determinent(m)
    if det!=0:
        return 1/det*adjugate(m)
    else:
        print("This matrix has no inverse as its determinent is zero")

def cross_product(a, b):
    # size = max(a.shape[0], a.shape[1])
    size = a.shape[0]
    m = np.ones((size, size), dtype=np.complex_)
    for i in range(size):
        m[1, i] = a[i]
        m[2, i] = b[i]
    values = []
    list = list_from_array(m)
    size = int(len(list) ** 0.5)
    if size == 3:
        for i in range(size):
            values.append((-1) ** i * list[i] * determinent(minor(list, size, i)))
        return values
    elif size == 2:
        return determinent2x2list(list)
    elif size < 2:
        return 0
    else:
        for i in range(size):
            values.append((-1) ** i * list[i] * determinent(minor(list, size, i)))
        return sum(values)


def trace(m):
    """Returns the Trace of a matrix

    Sums the main diagonal components of a square matrix
    """
    a = m.flat[::m.shape[0] + 1]
    return sum(a)


def inverse2x2(m):
    matrix = np.asarray([[m[0, 0], -m[0, 1]], [-m[1, 0], m[0, 0]]], dtype=np.complex_)
    factor = (1 / determinent(m))
    return factor, matrix


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


if (__name__ == '__main__'):
    # test = np.asarray([[2, -3, 1], [2, 0, -1], [1, 4, 5]], dtype=np.complex)
    # print(gauss_el([[2.0,1.0,-1.0,8.0],
    #                [-3.0,-1.0,2.0,-11.0],
    #                [-2.0,1.0,2.0,-3.0]]))
    test = np.asarray([[2, -3, 1, 1, 2], [0, -1, 1, 1, 4], [5, 1, 1, 1, 1], [1, 2, 3, 1, 2], [3, 4, 1, 6, 4]],
                      dtype=np.complex)
    test2 = np.asarray([[2, -3, 1, 1, 2], [0, -1, 1, 1, 4], [5, 1, 1, 1, 1], [1, 2, 3, 1, 2], [3, 4, 1, 6, 4]])
    print(test)
    print(test2)
    # print(len(test))
    print(upper_triangle(test))
    print(upper_triangle(test2))
    # print(determinent(test))
    # print(gauss_det(test))
    # print(gauss_det(test2))
    a = np.asarray([3, -3, 1], dtype=np.complex)
    b = np.asarray([4, 9, 2], dtype=np.complex)
    print(cross_product(a, b))
    print(transpose(a))
