import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTri


class ElementTriBubble(ElementH1):
    """The cubic bubble, i.e. a single interior DOF."""

    interior_dofs = 1
    maxdeg = 3
    dofnames = ['u']
    doflocs = np.array([[1. / 3., 1. / 3.]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = 27. * x * y * (1. - x - y)
            dphi = 27. * np.array([y * (1. - x - y) - x * y,
                                   x * (1. - x - y) - x * y])
        else:
            self._index_error()

        return phi, dphi
