r"""Minimal surface problem.

This example solves the nonlinear minimal surface problem using Newton's method.

"""

from skfem import *
from skfem.helpers import grad, dot
import numpy as np

m = MeshTri().refined(5)


@BilinearForm
def jacobian(u, v, p):
    w = p['prev']
    return (1 / np.sqrt(1 + dot(grad(w), grad(w))) * dot(grad(u), grad(v))
            -2 * dot(grad(u), grad(w)) * dot(grad(w), grad(v))
            / 2 / (1 + dot(grad(w), grad(w))) ** (3 / 2))


@LinearForm
def rhs(v, p):
    w = p['prev']
    return dot(grad(w), grad(v)) / np.sqrt(1 + dot(grad(w), grad(w)))


basis = Basis(m, ElementTriP1())

x = basis.zeros()

D = m.boundary_nodes()
x[D] = np.sin(np.pi * m.p[0, D])

for itr in range(100):
    w = basis.interpolate(x)
    J = asm(jacobian, basis, prev=w)
    F = asm(rhs, basis, prev=w)
    x_prev = x.copy()
    x += 0.7 * solve(*condense(J, -F, D=D))
    if np.linalg.norm(x - x_prev) < 1e-8:
        break
    if __name__ == "__main__":
        print(np.linalg.norm(x - x_prev))

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot3, show
    plot3(m, x)
    show()
