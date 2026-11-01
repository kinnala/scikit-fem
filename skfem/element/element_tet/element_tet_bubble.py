import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTet


class ElementTetBubble(ElementH1):
    """The quartic bubble, i.e. a single interior DOF."""

    interior_dofs = 1
    maxdeg = 4
    dofnames = ['u']
    doflocs = np.array([[.25, .25, .25]])
    refdom = RefTet

    def lbasis(self, X, i):
        x, y, z = X

        if i == 0:
            phi = 256. * x * y * z * (1. - x - y - z)
            dphi = 256. * np.array([y * z * (1. - x - y - z) - x * y * z,
                                    z * x * (1. - x - y - z) - x * y * z,
                                    x * y * (1. - x - y - z) - x * y * z])
        else:
            self._index_error()

        return phi, dphi
