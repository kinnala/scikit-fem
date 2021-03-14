import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefLine


class ElementLineP2(ElementH1):

    nodal_dofs = 1
    interior_dofs = 1
    maxdeg = 2
    dofnames = ['u', 'u']
    doflocs = np.array([[0.],
                        [1.],
                        [.5]])
    refdom = RefLine

    def lbasis(self, X, i):
        x = X[0]

        if i == 0:
            phi = 1 - 3 * x + 2 * x ** 2
            dphi = np.array([-3 + 4 * x])
        elif i == 1:
            phi = -x + 2 * x ** 2
            dphi = np.array([-1 + 4 * x])
        elif i == 2:
            phi = 4 * x - 4 * x ** 2
            dphi = np.array([4 - 8 * x])
        else:
            self._index_error()

        return phi, dphi
