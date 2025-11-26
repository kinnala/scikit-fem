import numpy as np

from ..element_hdiv import ElementHdiv
from ...refdom import RefHex


class ElementHexRT1(ElementHdiv):
    """First-order Raviart-Thomas for cube."""

    facet_dofs = 1
    maxdeg = 1
    dofnames = ['u_n']
    doflocs = np.array([[1.0, 0.5, 0.5],
                        [0.5, 0.5, 1.0],
                        [0.5, 1.0, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                        [0.0, 0.5, 0.5]])
    refdom = RefHex

    def lbasis(self, X, i):
        x, y, z = X

        if i == 0:
            phi = np.array([x, 0 * y, 0 * z])
            dphi = 1. + 0. * x
        elif i == 1:
            phi = np.array([0 * x, 0 * y, z])
            dphi = 1. + 0. * x
        elif i == 2:
            phi = np.array([0 * x, y, 0 * z])
            dphi = 1. + 0. * x
        elif i == 3:
            phi = np.array([0 * x, y - 1., 0 * z])
            dphi = 1. + 0. * x
        elif i == 4:
            phi = np.array([0 * x, 0 * y, z - 1.])
            dphi = 1. + 0. * x
        elif i == 5:
            phi = np.array([x - 1., 0 * y, 0 * z])
            dphi = 1. + 0. * x
        else:
            self._index_error()

        return phi, dphi
