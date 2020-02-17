import numpy as np

from .element_quad2 import ElementQuad2


class ElementQuadS2(ElementQuad2):
    interior_dofs = 0
    doflocs = ElementQuad2.doflocs[:-1]

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
