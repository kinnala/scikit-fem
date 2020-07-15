r"""Laplace with inhomogeneous boundary

Another simple modification of `ex01.py`, this time showing how
to impose coordinate-dependent Dirichlet conditions with
:func:`~skfem.utils.condense` and :attr:`~skfem.InteriorBasis.doflocs`.
The forcing term is suppressed for simplicity. The boundary values are
set as the real part :math:`x^2 - y^2` of an analytic complex function
:math:`(x + \mathrm i y)^2` which is harmonic and so that is the exact
solution through the domain.

This is checked quantitatively by computing the integral of the squared
magnitude of the gradient, by evaluating the quadratic form associated with the
laplacian at the solution; the exact value is 8/3.

This code would also work with :func:`~skfem.element.ElementTriP1`, in
which case, since the three nodes of the elements coincide with the
points of the mesh, the coordinate-dependent Dirichlet conditions
could be assigned more directly; however, the present method is
recommended as more general.

"""

from skfem import *
from skfem.models.poisson import laplace

import numpy as np

m = MeshTri()
m.refine(4)

e = ElementTriP2()
basis = InteriorBasis(m, e)

A = asm(laplace, basis)


def dirichlet(x, y):
    """return a harmonic function"""
    return ((x + 1.j * y) ** 2).real


boundary_basis = FacetBasis(m, e)
boundary_dofs = boundary_basis.find_dofs()['all'].all()

u = np.zeros(basis.N)
u[boundary_dofs] = L2_projection(dirichlet, boundary_basis, boundary_dofs)
u = solve(*condense(A, np.zeros_like(u), u, D=boundary_dofs))


if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show
    print('||grad u||**2 = {:f} (exact = 8/3 = {:f})'.format(u @ A @ u, 8/3))
    plot(basis, u)
    show()
