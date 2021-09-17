from ...refdom import RefTri
from ..element_h1 import ElementH1

import numpy as np


class ElementTriSkeletonP1(ElementH1):

    facet_dofs = 2
    maxdeg = 1
    dofnames = ['u', 'u']
    doflocs = np.array([[.0, .0],
                        [1., .0],
                        [1., .0],
                        [.0, 1.],
                        [.0, .0],
                        [.0, 1.]])
    refdom = RefTri

    def lbasis(self, X, i):

        if i in [0, 1, 2, 3, 4, 5]:
            return self.fbasis(X, i) * RefTri.on_facet(int(i / 2), X), 0. * X
        else:
            self._index_error()

    def fbasis(self, X, i):

        if i == 0:
            return 1. - X[0]
        elif i == 1:
            return X[0]
        elif i == 2:
            return X[0]
        elif i == 3:
            return X[1]
        elif i == 4:
            return 1. - X[1]
        elif i == 5:
            return X[1]
        else:
            self._index_error()
