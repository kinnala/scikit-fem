from skfem import *
import numpy as np
import matplotlib.pyplot as plt

"""
Solve the nonlinear minimal surface problem using automatic differentiation.
"""

m = MeshTri()
m.refine(5)

@nonlinear_form
def bilin(u, du, v, dv, w):
    import autograd.numpy as anp
    return 1.0/anp.sqrt(1.0 + du[0]**2 + du[1]**2)*(du[0]*dv[0] + du[1]*dv[1])

map = MappingAffine(m)
basis = InteriorBasis(m, ElementTriP1(), map, 2)

x = 0*m.p[0, :]
w, dw = basis.interpolate(x, derivative=True)

w = np.array([w, dw[0], dw[1]])
I = m.interior_nodes()
D = m.boundary_nodes()
x[D] = np.sin(np.pi*m.p[0, D])

for itr in range(10):
    J = asm(bilin, basis, w=w)
    F = asm(bilin.rhs, basis, w=w)
    x[I] = x[I] + solve(*condense(J, -F, I=I))
    print(np.linalg.norm(x))
    w, dw = basis.interpolate(x, derivative=True)
    w = np.array([w, dw[0], dw[1]])

m.plot3(x)
m.show()
