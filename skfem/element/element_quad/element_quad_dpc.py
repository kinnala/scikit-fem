from itertools import product

import numpy as np
from numpy.polynomial.legendre import legval, legder

from ..element_h1 import ElementH1
from ...refdom import RefQuad


class ElementQuadDPC(ElementH1):
    """Discontinuous element with polynomials of total degree :math:`p`.

    The basis consists of tensor products of Legendre polynomials,
    orthonormal over the reference element, with the total degree at most
    :math:`p`.  The first basis function is the constant one.  The space is
    defined on the reference element, i.e. it is not the full space of
    polynomials of total degree :math:`p` on non-affine elements.

    """

    refdom = RefQuad

    def __init__(self, p):
        if p < 0:
            raise ValueError("p < 0 not supported.")
        self.p = p
        self.maxdeg = p
        # multi-indices ordered by total degree
        self._ix = sorted(
            (ix for ix in product(range(p + 1), repeat=self.dim)
             if sum(ix) <= p),
            key=lambda ix: (sum(ix), tuple(-k for k in ix)),
        )
        self.interior_dofs = len(self._ix)
        self.dofnames = self.interior_dofs * ['u']
        self.doflocs = np.full((self.interior_dofs, self.dim), np.nan)

    @staticmethod
    def _legendre(x, n):
        """Orthonormal Legendre polynomial of degree n on [0, 1]."""
        c = np.zeros(n + 1)
        c[n] = np.sqrt(2. * n + 1.)
        return legval(2. * x - 1., c), 2. * legval(2. * x - 1., legder(c))

    def lbasis(self, X, i):
        if i < 0 or i >= self.interior_dofs:
            self._index_error()

        vals, ders = zip(*(self._legendre(X[j], n)
                           for j, n in enumerate(self._ix[i])))
        phi = np.prod(vals, axis=0)
        dphi = np.array([np.prod([ders[j] if j == k else vals[j]
                                  for j in range(self.dim)], axis=0)
                         for k in range(self.dim)])

        return phi, dphi
