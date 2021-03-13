import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTet


class ElementTetMini(ElementH1):

    nodal_dofs = 1
    interior_dofs = 1
    maxdeg = 4
    dofnames = ['u', 'NA']
    doflocs = np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [np.nan, np.nan, np.nan]])
    refdom = RefTet

    def lbasis(self, X, i):
        x, y, z = X

        if i == 0:
            phi = 1. - x - y - z
            dphi = np.array([-1. + 0. * x, -1. + 0. * x, -1. + 0. * x])
        elif i == 1:
            phi = x
            dphi = np.array([1. + 0. * x, 0. * x, 0. * x])
        elif i == 2:
            phi = y
            dphi = np.array([0. * x, 1. + 0. * x, 0. * x])
        elif i == 3:
            phi = z
            dphi = np.array([0. * x, 0. * x, 1. + 0. * x])
        elif i == 4:
            phi = 256. * x * y * z * (1. - x - y - z)
            dphi = 256. * np.array([y * z * (1. - x - y - z) - x * y * z,
                                    z * x * (1. - x - y - z) - x * y * z,
                                    x * y * (1. - x - y - z) - x * y * z])
        else:
            self._index_error()

        return phi, dphi
