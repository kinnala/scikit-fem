from skfem import *
from skfem.helpers import *
from skfem.models.poisson import unit_load
import numpy as np

m = (
    MeshTri.init_symmetric()
    .refined(4)
)

e = ElementTriHHJ() * ElementTriP2G()
basis = Basis(m, e)
ifbasis = [InteriorFacetBasis(m, e, side=i) for i in [0, 1]]


@BilinearForm
def bilinf(sig, u, tau, v, w):
    return ddot(sig, tau) - ddot(sig, dd(v)) - ddot(tau, dd(u))


@BilinearForm
def jump(sig, u, tau, v, w):
    return ddot(sig, prod(w.n, w.n)) * dot(grad(v), w.n)


@LinearForm
def linf(tau, v, w):
    return -1. * v


K = asm(bilinf, basis)
B = asm(jump, ifbasis)
f = asm(linf, basis)

D = basis.get_dofs().all(['u^2'])

x = solve(*condense(K + B + B.T, f, D=D))

(sig, sigbasis), (u, ubasis) = basis.split(x)

if __name__ == "__main__":
    ubasis.plot(u, colorbar=True, shading='gouraud').show()
