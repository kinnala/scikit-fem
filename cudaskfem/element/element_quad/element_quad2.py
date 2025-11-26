import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefQuad


class ElementQuad2(ElementH1):
    """Biquadratic element."""

    nodal_dofs = 1
    facet_dofs = 1
    interior_dofs = 1
    maxdeg = 4
    dofnames = ["u", "u", "u"]
    doflocs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                        [0.5, 0.0],
                        [1.0, 0.5],
                        [0.5, 1.0],
                        [0.0, 0.5],
                        [0.5, 0.5]])
    refdom = RefQuad

    def lbasis(self, X, i):
        X = 2 * X - 1
        x, y = X

        if i == 0:
            phi = 0.25 * (x ** 2 - x) * (y ** 2 - y)
            dphi = np.array([((-1 + 2 * x) * (-1 + y) * y) / 4.0,
                             ((-1 + x) * x * (-1 + 2 * y)) / 4.0])
        elif i == 1:
            phi = 0.25 * (x ** 2 + x) * (y ** 2 - y)
            dphi = np.array([((1 + 2 * x) * (-1 + y) * y) / 4.0,
                             (x * (1 + x) * (-1 + 2 * y)) / 4.0])
        elif i == 2:
            phi = 0.25 * (x ** 2 + x) * (y ** 2 + y)
            dphi = np.array([((1 + 2 * x) * y * (1 + y)) / 4.0,
                             (x * (1 + x) * (1 + 2 * y)) / 4.0])
        elif i == 3:
            phi = 0.25 * (x ** 2 - x) * (y ** 2 + y)
            dphi = np.array([((-1 + 2 * x) * y * (1 + y)) / 4.0,
                             ((-1 + x) * x * (1 + 2 * y)) / 4.0])
        elif i == 4:
            phi = 0.5 * (y ** 2 - y) * (1 - x ** 2)
            dphi = np.array([-(x * (-1 + y) * y),
                             -((-1 + x ** 2) * (-1 + 2 * y)) / 2.0])
        elif i == 5:
            phi = 0.5 * (x ** 2 + x) * (1 - y ** 2)
            dphi = np.array([-((1 + 2 * x) * (-1 + y ** 2)) / 2.0,
                             -(x * (1 + x) * y)])
        elif i == 6:
            phi = 0.5 * (y ** 2 + y) * (1 - x ** 2)
            dphi = np.array([-(x * y * (1 + y)),
                             -((-1 + x ** 2) * (1 + 2 * y)) / 2.0])
        elif i == 7:
            phi = 0.5 * (x ** 2 - x) * (1 - y ** 2)
            dphi = np.array([-((-1 + 2 * x) * (-1 + y ** 2)) / 2.0,
                             -((-1 + x) * x * y)])
        elif i == 8:
            phi = (1 - x ** 2) * (1 - y ** 2)
            dphi = np.array([2 * x * (-1 + y ** 2),
                             2 * (-1 + x ** 2) * y])
        else:
            self._index_error()

        return phi, 2 * dphi
