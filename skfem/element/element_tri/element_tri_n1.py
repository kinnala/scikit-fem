import numpy as np

from ..element_hcurl import ElementHcurl
from ...refdom import RefTri


class ElementTriN1(ElementHcurl):
    """The lowest order Nédélec element on a triangle."""

    facet_dofs = 1
    maxdeg = 1
    dofnames = ['u^t']
    doflocs = np.array([[.5, .0],
                        [.5, .5],
                        [.0, .5]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = np.array([y - 1., -x])
            dphi = -2. + 0. * x
        elif i == 1:
            phi = np.array([y, -x])
            dphi = -2. + 0. * x
        elif i == 2:
            phi = np.array([-y, -1. + x])
            dphi = 2. + 0. * x
        else:
            self._index_error()

        return phi, dphi
