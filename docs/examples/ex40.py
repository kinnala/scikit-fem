"""Hybridizable discontinuous Galerkin method.

This examples solves the Poisson equation with unit load using a technique
where the finite element basis is first discontinous across element edges and
then the continuity is recovered with the help of Lagrange multipliers defined
on the mesh skeleton (i.e. a "skeleton mesh" consisting only of the edges of
the original mesh).

As usual for these so-called "hybridizable" methods, the resulting system can
be condensed to the skeleton mesh only using Schur's complement.  However, this
is not done here as the example is meant to simply demonstrate the use of
finite elements defined on the mesh skeleton.

"""

from skfem import *
from skfem.helpers import grad, dot, jump
import numpy as np

m = MeshTri().refined(3)
e = ElementTriP1DG() * ElementTriSkeletonP1()
ibasis = Basis(m, e)
tbasis1 = InteriorFacetBasis(m, e, side=0)
tbasis2 = InteriorFacetBasis(m, e, side=1)
fbasis = FacetBasis(m, e)


@BilinearForm
def laplace(u, ut, v, vt, w):
    return dot(grad(u), grad(v))

@BilinearForm
def hdg(u, ut, v, vt, w):
    # outwards normal
    n = jump(w, w.n)
    return dot(n, grad(u)) * (vt - v) + dot(n, grad(v)) * (ut - u)\
        + 1e1 / w.h * (ut - u) * (vt - v)

@LinearForm
def load(v, vt, w):
    return 1. * v


A = asm(laplace, ibasis)
B = asm(hdg, [tbasis1, tbasis2, fbasis])
f = asm(load, ibasis)

y = solve(*condense(A + B, f, D=ibasis.get_dofs()))

(u1, _), (ut, skelebasis) = ibasis.split(y)

def visualize():
    from skfem.visuals.matplotlib import plot
    return plot(skelebasis,
                ut,
                Nrefs=4,
                colorbar=True,
                shading='gouraud')

if __name__ == '__main__':
    visualize().show()
