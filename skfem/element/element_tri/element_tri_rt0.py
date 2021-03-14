import numpy as np

from ..element_hdiv import ElementHdiv
from ...refdom import RefTri


class ElementTriRT0(ElementHdiv):

    facet_dofs = 1
    maxdeg = 1
    dofnames = ['u^n']
    doflocs = np.array([[.5, .0],
                        [.5, .5],
                        [.0, .5]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = np.array([x, y - 1.])
            dphi = 2. + 0. * x
        elif i == 1:
            phi = np.array([x, y])
            dphi = 2. + 0. * x
        elif i == 2:
            phi = np.array([x - 1., y])
            dphi = 2. + 0. * x
        else:
            self._index_error()

        return phi, dphi
