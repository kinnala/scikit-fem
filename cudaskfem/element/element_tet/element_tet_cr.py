import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTet


class ElementTetCR(ElementH1):
    """Nonconforming Crouzeix-Raviart element."""

    facet_dofs = 1
    maxdeg = 1
    dofnames = ['u']
    doflocs = np.array([[1 / 3, 1 / 3, 0.],
                        [1 / 3, 0., 1 / 3],
                        [0., 1 / 3, 1 / 3],
                        [1 / 3, 1 / 3, 1 / 3]])
    refdom = RefTet

    def lbasis(self, X, i):
        x, y, z = X

        if i == 0:
            phi = 1. - 3. * z
            dphi = np.array([0. * x, 0. * y, -3. + 0. * z])
        elif i == 1:
            phi = 1. - 3. * y
            dphi = np.array([0. * x, -3. + 0. * y, 0. * z])
        elif i == 2:
            phi = 1. - 3. * x
            dphi = np.array([-3. + 0. * x, 0. * y, 0. * z])
        elif i == 3:
            phi = 3. * x + 3. * y + 3. * z - 2.
            dphi = np.array([3. + 0. * x,
                             3. + 0. * y,
                             3. + 0. * z])
        else:
            self._index_error()

        return phi, dphi
