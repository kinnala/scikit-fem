from ...refdom import RefHex
from ..element_h1 import ElementH1

import numpy as np


class ElementHexSkeleton0(ElementH1):
    """Constant element for the mesh skeleton."""

    facet_dofs = 1
    maxdeg = 0
    dofnames = ['u']
    doflocs = np.array([[1.0, 0.5, 0.5],
                        [0.5, 0.5, 1.0],
                        [0.5, 1.0, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                        [0.0, 0.5, 0.5]])
    refdom = RefHex

    def lbasis(self, X, i):

        if i in range(6):
            return 1. * RefHex.on_facet(i, X), 0. * X
        else:
            self._index_error()
