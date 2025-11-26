import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefLine


class ElementLineMini(ElementH1):
    """The MINI element, i.e. linear element with additional bubble DOF."""

    nodal_dofs = 1
    interior_dofs = 1
    maxdeg = 2
    dofnames = ['u', 'NA']
    doflocs = np.array([[0.],
                        [1.],
                        [np.nan]])
    refdom = RefLine

    def lbasis(self, X, i):
        x = X[0]

        if i == 0:
            phi = 1. - x
            dphi = np.array([-1. + 0. * x])
        elif i == 1:
            phi = x
            dphi = np.array([1. + 0. * x])
        elif i == 2:
            phi = 4 * x * (1. - x)
            dphi = np.array([4. * (1. - 2. * x)])
        else:
            self._index_error()

        return phi, dphi
