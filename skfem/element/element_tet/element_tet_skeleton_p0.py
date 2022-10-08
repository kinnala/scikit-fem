from ...refdom import RefTet
from ..element_h1 import ElementH1

import numpy as np


class ElementTetSkeletonP0(ElementH1):
    """Constant element for the mesh skeleton."""

    facet_dofs = 1
    maxdeg = 0
    dofnames = ['u']
    doflocs = np.array([[.5, .5, .0],
                        [.5, .0, .5],
                        [.0, .5, .5],
                        [.5, .5, .5]])
    refdom = RefTet

    def lbasis(self, X, i):

        if i in [0, 1, 2, 3]:
            return 1. * RefTet.on_facet(i, X), 0. * X
        else:
            self._index_error()
