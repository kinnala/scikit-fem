"""Mixed Poisson equation and Raviart-Thomas basis on a MeshQuad"""

import numpy as np
from skfem import *
from skfem.helpers import dot, div


p = np.linspace(0, 1, 10)
m = MeshQuad.init_tensor(*(p,) * 2)

e = ElementQuadRT0() * ElementQuad0()
basis = Basis(m, e)


@BilinearForm
def bilinf(sigma, u, tau, v, w):
    return dot(sigma, tau) + div(sigma) * v + div(tau) * u


@LinearForm
def linf(tau, v, w):
    return - 1. * v


A = asm(bilinf, basis)
b = asm(linf, basis)
x = solve(A, b)

_, (u, __) = basis.split(x)


if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show
    plot(m, u)
    show()
