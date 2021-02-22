import warnings
from typing import Type

import numpy as np
from numpy.polynomial.legendre import Legendre

from ..element_h1 import ElementH1
from ...refdom import Refdom, RefLine


class ElementLinePp(ElementH1):

    nodal_dofs = 1
    refdom: Type[Refdom] = RefLine

    def __init__(self, p):
        if p < 1:
            raise ValueError("p < 1 not supported.")
        if p < 3:
            warnings.warn(("Consider using ElementLineP{} instead "
                           "of ElementLinePp.").format(p - 1))
        self.interior_dofs = p - 1
        self.maxdeg = p
        self.dofnames = ['u'] + (p - 1) * ['u']
        self.doflocs = np.array([[0., 1.] + [np.nan] * (p - 1)]).T
        self.P = np.zeros((0, 0))
        self.dP = np.zeros((0, 0, 1))
        self.p = p

    @staticmethod
    def _reval_legendre(y, p):
        """Re-evaluate Legendre polynomials."""
        P = np.zeros((p + 1,) + y.shape)
        dP = np.zeros((p + 1, 1) + y.shape)

        P[0] = 1. - y
        P[1] = y
        dP[0][0] = -1. + 0. * y
        dP[1][0] = 1. + 0. * y

        x = 2. * y - 1.
        for n in range(2, p + 1):
            c = np.zeros(n)
            c[n - 1] = 1.
            s = Legendre(c).integ(lbnd=-1)
            scale = np.sqrt((2. * n - 1.) / 2.)
            P[n] = s(x) * scale
            dP[n][0] = 2 * s.deriv()(x) * scale

        return P, dP

    def lbasis(self, X, i):

        if self.P.shape[1] != X.shape[1]:
            self.P, self.dP = self._reval_legendre(X[0, :], self.p)

        return self.P[i], self.dP[i]
