from skfem import *
from skfem.weakforms import laplace

"""
High-order plotting test.
"""

m = MeshTri()
m.refine(2)

e = ElementLocalTriP2()
a = AssemblerLocal(m, e)

K = a.iasm(laplace)
f = a.iasm(lambda v: 1*v)

_, D = a.essential_bc()

x = direct(K, f, D=D)

M, X = a.refinterp(x, 3)

M.plot(X, smooth=True, edgecolors='')
M.show()
