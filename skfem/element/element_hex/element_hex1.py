import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefHex


class ElementHex1(ElementH1):

    nodal_dofs = 1
    maxdeg = 3
    dofnames = ['u']
    doflocs = np.array([[1., 1., 1.],
                        [1., 1., 0.],
                        [1., 0., 1.],
                        [0., 1., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [0., 0., 0.]])
    refdom = RefHex

    def lbasis(self, X, i):
        x, y, z = X

        if i == 0:
            phi = x * y * z
            dphi = np.array([y * z,
                             x * z,
                             x * y])
        elif i == 1:
            phi = x * y * (1 - z)
            dphi = np.array([y * (1 - z),
                             x * (1 - z),
                             -x * y])
        elif i == 2:
            phi = x * (1 - y) * z
            dphi = np.array([(1 - y) * z,
                             -x * z,
                             x * (1 - y)])
        elif i == 3:
            phi = (1 - x) * y * z
            dphi = np.array([-y * z,
                             (1 - x) * z,
                             (1 - x) * y])
        elif i == 4:
            phi = x * (1 - y) * (1 - z)
            dphi = np.array([(1 - y) * (1 - z),
                             -x * (1 - z),
                             -x * (1 - y)])
        elif i == 5:
            phi = (1 - x) * y * (1 - z)
            dphi = np.array([-y * (1 - z),
                             (1 - x) * (1 - z),
                             -(1 - x) * y])
        elif i == 6:
            phi = (1 - x) * (1 - y) * z
            dphi = np.array([-(1 - y) * z,
                             -(1 - x) * z,
                             (1 - x) * (1 - y)])
        elif i == 7:
            phi = (1 - x) * (1 - y) * (1 - z)
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        else:
            self._index_error()

        return phi, dphi
