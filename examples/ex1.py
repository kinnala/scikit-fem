"""
Solve the problem -∇²u = 1 with zero boundary conditions
on a unit square.
"""

from skfem import *

m = MeshTri()
m.refine(4)

e = ElementTriP1()
map = MappingAffine(m)
basis = InteriorBasis(m, e, map, 2)

@bilinear_form
def laplace(u, du, v, dv, w):
    return du[0]*dv[0] + du[1]*dv[1]

@linear_form
def load(v, dv, w):
    return 1.0*v

A = asm(laplace, basis)
b = asm(load, basis)

I = m.interior_nodes()

x = 0*b
x[I] = solve(*condense(A, b, I=I))

m.plot3(x)
m.show()
