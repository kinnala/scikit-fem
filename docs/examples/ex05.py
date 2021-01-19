r"""Integral condition.

This short example demonstrates the implementation of an integral boundary
 condition

.. math::
   
   \int_\Gamma \nabla u \cdot \boldsymbol{n} \, \mathrm{d}s = 1

on a part of the boundary of the domain :math:`\Gamma \subset \partial \Omega`
 for the Laplace operator.  In this example, :math:`\Gamma` is the right
 boundary of the unit square and the solution satisfies :math:`u=0` on the
 bottom boundary and :math:`\nabla u \cdot \boldsymbol{n} = 0` on the rest of
 the boundaries.  The constraint is introduced via a Lagrange multiplier leading
 to a saddle point system.

"""

from skfem import *
from skfem.helpers import dot, grad
from skfem.models.poisson import laplace

m = MeshTri().refined(5)

e = ElementTriP1()

ib = InteriorBasis(m, e)
fb = FacetBasis(m, e)


@BilinearForm
def facetbilinf(u, v, w):
    n = w.n
    x = w.x
    return -dot(grad(u), n) * v * (x[0] == 1.0)


@LinearForm
def facetlinf(v, w):
    n = w.n
    x = w.x
    return -dot(grad(v), n) * (x[0] == 1.0)


A = asm(laplace, ib)
B = asm(facetbilinf, fb)

b = asm(facetlinf, fb)

D = ib.find_dofs({'plate': m.facets_satisfying(lambda x: (x[1] == 0.0))})
I = ib.complement_dofs(D)

import scipy.sparse
b = scipy.sparse.csr_matrix(b)
K = scipy.sparse.bmat([[A+B, b.T], [b, None]], 'csr')

import numpy as np
f = np.concatenate((np.zeros(A.shape[0]), -1.0*np.ones(1)))

I = np.append(I, K.shape[0] - 1)

x = solve(*condense(K, f, I=I))

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import plot, savefig
    plot(m, x[:-1], colorbar=True)
    savefig(splitext(argv[0])[0] + '_solution.png')
