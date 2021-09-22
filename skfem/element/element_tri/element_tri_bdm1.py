import numpy as np

from ..element_hdiv import ElementHdiv
from ...refdom import RefTri


# two Gauss points used in the definition of the basis
s_1 = .5 - np.sqrt(3) / 6.
s_2 = .5 + np.sqrt(3) / 6.


class ElementTriBDM1(ElementHdiv):
    """The lowest order Brezzi-Douglas-Marini element."""

    facet_dofs = 2
    maxdeg = 2
    dofnames = ['u^n', 'u^n']
    doflocs = np.array([[s_1, .0],
                        [s_2, .0],
                        [s_2, 1. - s_2],
                        [s_1, 1. - s_1],
                        [.0, s_1],
                        [.0, s_2]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        # absolute value of dphi must be the same!
        if i == 0:
            phi = -1. / (s_1 - s_2) * np.array([(s_2 - 1) * x,
                                                x + s_2 * y - s_2])
            dphi = -1. / (s_1 - s_2) * (2. * s_2 - 1) + 0. * x
        elif i == 1:
            phi = -1. / (s_2 - s_1) * np.array([(s_1 - 1) * x,
                                                x + s_1 * y - s_1])
            dphi = -1. / (s_2 - s_1) * (2. * s_1 - 1) + 0. * x
        elif i == 2:
            phi = 1. / (s_2 - s_1) * np.array([s_2 * x,
                                               (s_2 - 1) * y])
            dphi = 1. / (s_2 - s_1) * (2. * s_2 - 1) + 0. * x
        elif i == 3:
            phi = 1. / (s_1 - s_2) * np.array([s_1 * x,
                                               (s_1 - 1) * y])
            dphi = 1. / (s_1 - s_2) * (2. * s_1 - 1) + 0. * x
        elif i == 4:
            phi = 1. / (s_2 - s_1) * np.array([s_2 * x + y - s_2,
                                               (s_2 - 1) * y])
            dphi = 1. / (s_2 - s_1) * (2. * s_2 - 1) + 0. * x
        elif i == 5:
            phi = 1. / (s_1 - s_2) * np.array([s_1 * x + y - s_1,
                                               (s_1 - 1) * y])
            dphi = 1. / (s_1 - s_2) * (2. * s_1 - 1) + 0. * x
        else:
            self._index_error()

        return phi, dphi
