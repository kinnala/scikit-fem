"""Nonlinear Poisson equation.

This example solves the nonlinear minimal surface problem using a Jacobian-free Newton-Krylov method.

"""

from skfem import *
from skfem.helpers import grad, dot
import numpy as np
from scipy.optimize import root

m = MeshTri()
m.refine(5)


@LinearForm
def rhs(v, w):
    return dot(grad(w['w']), grad(v)) / np.sqrt(1 + dot(grad(w['w']), grad(w['w'])))


basis = InteriorBasis(m, ElementTriP1())

x = np.zeros(basis.N)

I = m.interior_nodes()
D = m.boundary_nodes()
x[D] = np.sin(np.pi * m.p[0, D]) 

def residual(u: np.ndarray) -> np.ndarray:
    u_full = np.empty_like(x)
    u_full[I] = u
    u_full[D] = x[D]

    return asm(rhs, basis, w=basis.interpolate(u_full))[I]

sol = root(residual, x[I], method='krylov', options={'disp': True})
x[I] = sol.x 


if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot3, show
    plot3(m, x)
    show()
