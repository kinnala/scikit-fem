import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefQuad


class ElementQuad0(ElementH1):

    interior_dofs = 1
    maxdeg = 0
    dofnames = ['u']
    doflocs = np.array([[.5, .5]])
    refdom = RefQuad

    def lbasis(self, X, i):
        if i == 0:
            return np.ones(X.shape[1:]), np.zeros_like(X)
        else:
            self._index_error()
