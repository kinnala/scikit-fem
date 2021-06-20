import numpy as np

from ..element_hcurl import ElementHcurl
from ...refdom import RefTet


class ElementTetN0(ElementHcurl):
    """The lowest order Nédélec element."""

    edge_dofs = 1
    maxdeg = 1
    dofnames = ['u^t']
    doflocs = np.array([[.5, 0., 0.],
                        [.5, .5, 0.],
                        [0., .5, 0.],
                        [0., 0., .5],
                        [.5, 0., .5],
                        [0., .5, .5]])
    refdom = RefTet

    def lbasis(self, X, i):
        x, y, z = X

        if i == 0:
            phi = np.array([1 - z - y, x, x])
            dphi = np.array([0 * x,
                             -2 + 0 * x,
                             2 + 0 * x])
        elif i == 1:
            phi = np.array([-y, x, 0 * z])
            dphi = np.array([0 * x,
                             0 * x,
                             2 + 0 * x])
        elif i == 2:
            phi = np.array([y, 1 - z - x, y])
            dphi = np.array([2 + 0 * x,
                             0 * x,
                             -2 + 0 * x])
        elif i == 3:
            phi = np.array([z, z, 1 - x - y])
            dphi = np.array([-2 + 0 * x,
                             2 + 0 * x,
                             0 * x])
        elif i == 4:
            phi = np.array([-z, 0 * y, x])
            dphi = np.array([0 * x,
                             -2 + 0 * x,
                             0 * x])
        elif i == 5:
            phi = np.array([0 * x, -z, y])
            dphi = np.array([2 + 0 * x,
                             0 * x,
                             0 * x])
        else:
            self._index_error()

        return phi, dphi
