import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTri


class ElementTriCR(ElementH1):

    facet_dofs = 1
    maxdeg = 1
    dofnames = ['u']
    doflocs = np.array([[.5, 0.],
                        [.5, .5],
                        [0., .5]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = 1. - 2. * y
            dphi = np.array([0. * x, -2. + 0. * y])
        elif i == 1:
            phi = 2. * x + 2. * y - 1.
            dphi = np.array([2. + 0. * x, 2. + 0. * y])
        elif i == 2:
            phi = 1. - 2. * x
            dphi = np.array([-2. + 0. * x, 0. * x])
        else:
            self._index_error()

        return phi, dphi
