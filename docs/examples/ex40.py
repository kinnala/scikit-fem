"""# Hybridizable discontinuous Galerkin method

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
import numpy as np

m = MeshTri().refined(3)
e = ElementTriP1DG() * ElementTriSkeletonP1()
ibasis = Basis(m, e)
tbasis = [
    *[InteriorFacetBasis(m, e, side=i) for i in [0, 1]],
    FacetBasis(m, e)
]

@BilinearForm
def laplace(u, ut, v, vt, w):
    from skfem.helpers import grad, dot
    return dot(grad(u), grad(v))

@BilinearForm
def hdg(u, ut, v, vt, w):
    from skfem.helpers import grad, dot, jump
    # outwards normal
    n = jump(w, w.n)
    return dot(n, grad(u)) * (vt - v) + dot(n, grad(v)) * (ut - u)\
        + 1e1 / w.h * (ut - u) * (vt - v)

@LinearForm
def load(v, vt, w):
    return 1. * v

A = asm(laplace, ibasis) + asm(hdg, tbasis)
f = asm(load, ibasis)

y = solve(*condense(A, f, D=ibasis.get_dofs()))

(u1, ibasis1), (u2, ibasis2) = ibasis.split(y)

def visualize():
    from skfem.visuals.matplotlib import plot
    return plot(ibasis2,
                u2,
                Nrefs=4,
                colorbar=True,
                shading='gouraud')

if __name__ == '__main__':
    visualize().show()
