"""Hybridizable discontinuous Galerkin method.

This example solves the Poisson equation with unit load using a technique
where the finite element basis is first discontinuous across element edges and
then the continuity is recovered with the help of Lagrange multipliers defined
on the mesh skeleton, i.e. a mesh consisting only of the edges of the original
mesh.  The pair of bases, the traces of the discontinuous basis functions and
the basis on the mesh skeleton, is given by
:class:`~skfem.assembly.SkeletonBasis` which evaluates them over the edges of
each element.

As usual for these so-called "hybridizable" methods, the discontinuous unknowns
are eliminated element-by-element (static condensation) so that the resulting
linear system is defined on the mesh skeleton only.

"""

from skfem import *
from skfem.helpers import grad, dot

m = MeshTri().refined(3)
basis = Basis(m, ElementTriP1DG())
sbasis = basis.skeleton(ElementLineP1DG())


@BilinearForm
def laplace(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def hdg(u, ut, v, vt, w):
    # w.n is the outward normal of each element
    return (dot(w.n, grad(u)) * (vt - v) + dot(w.n, grad(v)) * (ut - u)
            + 1e1 / w.h * (ut - u) * (vt - v))


@LinearForm
def load(v, w):
    return 1. * v


# blocks of the discontinuous (u) and the skeleton (ut) unknowns
(Ah, B), (C, D) = hdg.elemental_blocks(sbasis)
lap = laplace.elemental(basis)

# the block of u is block diagonal: invert element-by-element
Ainv = lap.fromlocal(lap.tolocal() + Ah.tolocal(sbasis)).inverse().tocsr()
B, C, D = B.tocsr(), C.tocsr(), D.tocsr()
f = load.assemble(basis)

# solve for ut on the mesh skeleton; then recover u element-by-element
ut = solve(*condense(D - C @ Ainv @ B,
                     -C @ (Ainv @ f),
                     D=sbasis.get_dofs()))
u = Ainv @ (f - B @ ut)


def visualize():
    from skfem.visuals.web import plot
    ubasis, utbasis = sbasis.split_bases()
    return plot(utbasis, ut)


if __name__ == '__main__':
    visualize().show()
