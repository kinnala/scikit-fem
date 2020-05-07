"""Nonlinear Poisson equation.

This example solves the nonlinear minimal surface problem using Newton's method.

"""

from skfem import *
from skfem.helpers import grad, dot, trace
import numpy as np

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

for itr in range(100):
    w = basis.interpolate(x)
    J = asm(jacobian, basis, w=w)
    F = asm(rhs, basis, w=w)
    x_prev = x.copy()
    x += 0.7 * solve(*condense(J, -F, I=I))
    if np.linalg.norm(x - x_prev) < 1e-8:
        break
    if __name__ == "__main__":
        print(np.linalg.norm(x - x_prev))

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot3, show
    plot3(m, x)
    show()
