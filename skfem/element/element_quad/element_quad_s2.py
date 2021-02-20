import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefQuad


class ElementQuadS2(ElementH1):

    nodal_dofs = 1
    facet_dofs = 1
    maxdeg = 3
    dofnames = ["u", "u"]
    doflocs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                        [0.5, 0.0],
                        [1.0, 0.5],
                        [0.5, 1.0],
                        [0.0, 0.5]])
    refdom = RefQuad

    def lbasis(self, X, i):
        X = 2. * X - 1.
        x, y = X

        if i == 0:
            phi = -(1 - x) * (1 - y) * (x + y + 1) / 4
            dphi = np.array([(1 - y) * (2 * x + y),
                             (1 - x) * (x + 2 * y)]) / 4
        elif i == 1:
            phi = (1 + x) * (1 - y) * (x - y - 1) / 4
            dphi = np.array([(y - 1) * (y - 2 * x),
                             (1 + x) * (2 * y - x)]) / 4
        elif i == 2:
            phi = (1 + x) * (1 + y) * (x + y - 1) / 4
            dphi = np.array([(1 + y) * (2 * x + y),
                             (1 + x) * (x + 2 * y)]) / 4
        elif i == 3:
            phi = (1 - x) * (1 + y) * (y - x - 1) / 4
            dphi = np.array([(1 + y) * (2 * x - y),
                             (x - 1) * (x - 2 * y)]) / 4
        elif i == 4:
            phi = (1 - x**2) * (1 - y) / 2
            dphi = np.array([x * (y - 1),
                             (x**2 - 1) / 2])
        elif i == 5:
            phi = (1 - y**2) * (1 + x) / 2
            dphi = np.array([(1 - y**2) / 2,
                             -(1 + x) * y])
        elif i == 6:
            phi = (1 - x**2) * (1 + y) / 2
            dphi = np.array([-x * (y + 1),
                             (1 - x**2) / 2])
        elif i == 7:
            phi = (1 - y**2) * (1 - x) / 2
            dphi = np.array([(y**2 - 1) / 2,
                             (x - 1) * y])
        else:
            self._index_error()

        return phi, 2 * dphi
