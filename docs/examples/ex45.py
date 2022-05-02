r"""Automatic differentiation.

Warning: This example uses experimental features.

This example solves the minimal surface problem from example 10 using automatic
differentation in order to derive the tangent system for Newton's method.

"""
from skfem import *
from skfem.experimental.autodiff import NonlinearForm
from skfem.experimental.autodiff.helpers import grad, dot
import numpy as np
import autograd.numpy as anp

m = MeshTri().refined(5)


@NonlinearForm(hessian=True)
def energy(u, _):
    return anp.sqrt(1. + dot(grad(u), grad(u)))


basis = Basis(m, ElementTriP1())

D = m.boundary_nodes()
x = np.sin(np.pi * m.p[0])

for itr in range(100):
    J, F = energy.assemble(x, basis)
    x_prev = x.copy()
    x += solve(*condense(J, -F, D=D))
    res = np.linalg.norm(x - x_prev)
    if res < 1e-8:
        break
    if __name__ == "__main__":
        print(res)

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot3, show
    plot3(m, x)
    show()

