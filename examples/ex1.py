from skfem import *
from skfem.weakforms import laplace

m = MeshTri()
for itr in range(4):
    m.refine()

e = ElementLocalTriP1()
a = AssemblerLocal(m, e)

K = a.iasm(laplace)
f = a.iasm(lambda v: 1*v)

D = m.boundary_nodes()

x = direct(K, f, D=D)

m.plot3(x)
m.show()
