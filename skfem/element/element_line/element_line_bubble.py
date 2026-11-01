import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefLine


class ElementLineBubble(ElementH1):
    """The quadratic bubble, i.e. a single interior DOF."""

    interior_dofs = 1
    maxdeg = 2
    dofnames = ['u']
    doflocs = np.array([[.5]])
    refdom = RefLine

    def lbasis(self, X, i):
        x = X[0]

        if i == 0:
            phi = 4. * x * (1. - x)
            dphi = np.array([4. * (1. - 2. * x)])
        else:
            self._index_error()

        return phi, dphi
