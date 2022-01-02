r"""Laplace with inhomogeneous boundary

Another simple modification of `ex01.py`, this time showing how
to impose coordinate-dependent Dirichlet conditions with
:func:`~skfem.utils.condense` and :attr:`~skfem.CellBasis.doflocs`.
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

m = MeshTri().refined(4)

e = ElementTriP2()
basis = Basis(m, e)

A = asm(laplace, basis)


def dirichlet(x):
    """return a harmonic function"""
    return ((x[0] + 1.j * x[1]) ** 2).real


u = basis.project(dirichlet)
u = solve(*condense(A, x=u, D=basis.get_dofs()))


def visualize():
    from skfem.visuals.matplotlib import plot, show
    return plot(basis,
                u,
                shading='gouraud',
                colorbar=True,
                levels=5)


if __name__ == "__main__":
    print('||grad u||**2 = {:f} (exact = 8/3 = {:f})'
          .format(u @ A @ u, 8/3))
    visualize().show()
