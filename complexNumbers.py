

# c = Complex(1, 2)
# print(c.real)
# print(c.imag)
a = np.complex(1, 2)
# print(a)
b = 3 + 4j
c = 1j
d = 3
# print(b)
# a = a.conjugate()
# print(a)
# b = b.conjugate()
# print(b)
# test1 = np.zeros((2,2),dtype=np.complex_)
# test1[0,0]=a
# test1[0,1]=b
# test1[1,0]=c
# test1[1,1]=d
test1 = np.asarray([[a, b], [c, d], [a, c]])
print(test1)
rows = test1.shape[0]
columns = test1.shape[1]
newShape = (columns, rows)
t = np.zeros(newShape, dtype=np.complex)
for i in range(rows):
    for x in range(columns):
        # conjugate = test1[i,x].conj()
        # t[i, x] = complex(conjugate.real,conjugate.imag)
        t[i, x] = test1[i, x].conj()
print(t)
