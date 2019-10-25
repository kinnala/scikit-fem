import numpy as np
from skfem import *

m = MeshLine(np.linspace(0, 1, 10))

e = ElementLineP1()
basis = InteriorBasis(m, e)

@bilinear_form
def laplace(u, du, v, dv, w):
    return du[0]*dv[0]

@linear_form
def load(v, dv, w):
    return 1.0*v

A = asm(laplace, basis)
b = asm(load, basis)

D = basis.get_dofs().all()

x = 0*b
x = solve(*condense(A, b, D=D))

if __name__ == "__main__":
    m.plot(x)
    m.show()
