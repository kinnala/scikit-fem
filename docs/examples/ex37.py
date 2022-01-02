"""# Mixed Poisson equation and Raviart-Thomas basis

"""
import numpy as np
from skfem import *
from skfem.helpers import dot, div


p = np.linspace(0, 1, 10)
m = MeshTet.init_tensor(*(p,) * 3)

e = ElementTetRT0() * ElementTetP0()
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

(sigma, rtbasis), (u, ubasis) = basis.split(x)
M, X = ubasis.refinterp(u, Nrefs=0)

if __name__ == "__main__":
    M.save('ex37.vtk', {'sol': X})
