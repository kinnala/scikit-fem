"""Nonlinear Poisson equation.

This example solves the nonlinear minimal surface problem using a Newton-Krylov method.

"""

from skfem import *
from skfem.helpers import grad, dot
from skfem.models.poisson import laplace
import numpy as np
from scipy.optimize import root

m = MeshTri()
m.refine(5)


@BilinearForm
def jacobian(u, v, w):
    return (1 / np.sqrt(1 + dot(grad(w['w']), grad(w['w']))) * dot(grad(u), grad(v))
            -2 * dot(grad(u), grad(w['w'])) * dot(grad(w['w']), grad(v))
            / 2 / (1 + dot(grad(w['w']), grad(w['w'])))**(3/2))


@LinearForm
def rhs(v, w):
    return dot(grad(w['w']), grad(v)) / np.sqrt(1 + dot(grad(w['w']), grad(w['w'])))


basis = InteriorBasis(m, ElementTriP1())

x = np.zeros(basis.N)

I = m.interior_nodes()
D = m.boundary_nodes()
x[D] = np.sin(np.pi * m.p[0, D]) 

def residual(u: np.ndarray) -> np.ndarray:
    res = asm(rhs, basis, w=basis.interpolate(u))
    res[D] = 0.
    return res

M = build_pc_ilu(asm(jacobian, basis, w=basis.interpolate(x)))

sol = root(residual, x, method='krylov',
           options={'disp': True,
                    'jac_options': {
                        'inner_M': M}})

print(sol)
x = sol.x 


if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot3, show
    plot3(m, x)
    show()
