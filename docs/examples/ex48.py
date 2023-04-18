"""Solve :math:`\Delta^2 u = 1` using HHJ element."""
from skfem import *
from skfem.helpers import *
import numpy as np

m = MeshTri.init_sqsymmetric().refined(4)

e = ElementTriHHJ1() * ElementTriP2G()
#e = ElementTriHHJ0() * ElementTriP1G()
basis = Basis(m, e)
fbasis = basis.boundary()
ifbasis0 = InteriorFacetBasis(m, e, side=0)
ifbasis1 = InteriorFacetBasis(m, e, side=1)


@BilinearForm
def bilinf(sig, u, tau, v, w):
    return ddot(sig, tau) - ddot(sig, dd(v)) - ddot(tau, dd(u))


@BilinearForm
def jump(sig, u, tau, v, w):
    return (ddot(sig, prod(w.n, w.n)) * dot(grad(v), w.n)
            + ddot(tau, prod(w.n, w.n)) * dot(grad(u), w.n))


@LinearForm
def linf(tau, v, w):
    return -1. * v


K = asm(bilinf, basis)
B1 = asm(jump, ifbasis0)
B2 = asm(jump, ifbasis1)
B3 = asm(jump, fbasis)
f = asm(linf, basis)

D = basis.get_dofs().all(['u^2'])

x = solve(*condense(K + (B1 - B2 + B3), f, D=D))

(sig, sigbasis), (u, ubasis) = basis.split(x)

if __name__ == "__main__":
    basis0 = basis.with_element(ElementDG(ElementTriP1()))

    for itr in range(2):
        for jtr in range(2):
            sig0 = basis0.project(sigbasis.interpolate(sig)[itr, jtr])
            basis0.plot(sig0, colorbar=True)

    ubasis.plot(u, colorbar=True).show()
