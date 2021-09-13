from ...refdom import RefTri
from ..element_h1 import ElementH1

import numpy as np


class ElementTriSkeletonP0(ElementH1):

    facet_dofs = 1
    maxdeg = 0
    dofnames = ['u']
    doflocs = np.array([[.5, .0],
                        [.5, .5],
                        [.0, .5]])
    refdom = RefTri

    def lbasis(self, X, i):

        if i in [0, 1, 2]:
            return RefTri.on_facet(i, X), 0. * X
        else:
            self._index_error()
