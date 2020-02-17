import numpy as np
from ..element_h1 import ElementH1


class ElementLinePp(ElementH1):
    nodal_dofs = 1
    dim = 1

    def __init__(self, p):
        self.interior_dofs = p - 1
        self.maxdeg = p
        self.dofnames = ['u'] + (p - 1) * ['u']
        self.doflocs = np.array([[0., 1.] + [.5] * (p - 1)]).T
        self.P = np.zeros((0, 0))
        self.dP = np.zeros((0, 0, 1))
        self.p = p

    def lbasis(self, X, i):

        if self.P.shape[1] != X.shape[1]:
            # re-evaluate Legendre polynomials if needed
            from numpy.polynomial.legendre import Legendre

            y = X[0, :]
            self.P = np.zeros((self.p + 1, len(y)))
            self.dP = np.zeros((self.p + 1, 1, len(y)))

            self.P[0] = 1. - y
            self.P[1] = y
            self.dP[0][0] = -1. + 0. * y
            self.dP[1][0] = 1. + 0. * y

            x = 2. * y - 1.
            for n in range(2, self.p + 1):
                c = np.zeros(n)
                c[n - 1] = 1.
                s = Legendre(c).integ(lbnd=-1)
                scale = np.sqrt((2. * n - 1.) / 2.)
                self.P[n] = s(x) * scale
                self.dP[n][0] = 2 * s.deriv()(x) * scale

        return self.P[i], self.dP[i]
