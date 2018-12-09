"""
Author: kinnala

Solve the nonlinear minimal surface problem using Newton's method.

"""
from skfem import *
import numpy as np
import matplotlib.pyplot as plt

m = MeshTri()
m.refine(5)

@bilinear_form
def jacobian(u, du, v, dv, w):
    w, dw = w.w, w.dw
    return 1.0/np.sqrt(1.0 + dw[0]**2 + dw[1]**2)*(du[0]*dv[0] + du[1]*dv[1])\
           -(2.0*du[1]*dw[1] + 2.0*du[0]*dw[0])*(dw[1]*dv[1] + dw[0]*dv[0])\
           /(2.0*(1 + dw[1]**2 + dw[0]**2)**(3./2.))

@linear_form
def rhs(v, dv, w):
    w, dw = w.w, w.dw
    return 1.0/np.sqrt(1.0 + dw[0]**2 + dw[1]**2)*(dw[0]*dv[0] + dw[1]*dv[1])

map = MappingAffine(m)
basis = InteriorBasis(m, ElementTriP1(), map, 2)

x = 0*m.p[0, :]

I = m.interior_nodes()
D = m.boundary_nodes()
x[D] = np.sin(np.pi*m.p[0, D]) 

for itr in range(100):
    import copy
    w = basis.interpolate(x)
    J = asm(jacobian, basis, w=w)
    F = asm(rhs, basis, w=w)
    xprev = copy.deepcopy(x)
    x[I] = 0.3*x[I] + 0.7*solve(*condense(J, -F, I=I))
    if np.linalg.norm(x - xprev) < 1e-8:
        break
    if __name__ == "__main__":
        print(np.linalg.norm(x - xprev))

if __name__ == "__main__":
    m.plot3(x)
    m.show()
