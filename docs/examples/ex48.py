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
ifbasis0 = InteriorFacetBasis(m, e, side=0)
ifbasis1 = InteriorFacetBasis(m, e, side=1)


@BilinearForm
def bilinf(sig, u, tau, v, w):
    # from skfem.helpers import dd, ddot, trace, eye
    # d = 0.1
    # E = 200e9
    # nu = 0.3

    # def C(T):
    #     return E / (1 + nu) * (T + nu / (1 - nu) * eye(trace(T), 2))

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
f = asm(linf, basis)

D = basis.get_dofs().all(['u^2'])

x = solve(*condense(K + B1 + B2, f, D=D))

(sig, sigbasis), (u, ubasis) = basis.split(x)

if __name__ == "__main__":
    ubasis.plot(u, colorbar=True, shading='gouraud').show()
