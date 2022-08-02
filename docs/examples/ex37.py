"""Mixed Poisson equation and Raviart-Thomas basis"""

import numpy as np
from skfem import *
from skfem.helpers import dot, div


p = np.linspace(0, 1, 10)
m = MeshTet.init_tensor(*(p,) * 3)

e = ElementTetRT1() * ElementTetP0()
basis = Basis(m, e)
fbasis = basis.boundary()


@BilinearForm
def bilinf(sigma, u, tau, v, w):
    return dot(sigma, tau) + div(sigma) * v + div(tau) * u


@LinearForm
def linf(tau, v, w):
    return (dot(tau, w.n)
            * np.sin(np.pi * w.x[1])
            * np.sin(np.pi * w.x[2])
            * np.isclose(w.x[0], 1))


A = asm(bilinf, basis)
b = asm(linf, fbasis)
x = solve(A, b)

(sigma, rtbasis), (u, ubasis) = basis.split(x)
M, X = ubasis.refinterp(u, Nrefs=0)

if __name__ == "__main__":
    M.save('ex37.vtk', {'sol': X})
