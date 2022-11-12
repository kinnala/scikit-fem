import numpy as np

from ..element_hcurl import ElementHcurl
from ...refdom import RefQuad


class ElementQuadN1(ElementHcurl):
    """The lowest order Nédélec element."""

    facet_dofs = 1
    maxdeg = 1
    dofnames = ["u^t"]
    doflocs = np.array([[0.5, 0.0],
                        [1.0, 0.5],
                        [0.5, 1.0],
                        [0.0, 0.5]])
    refdom = RefQuad

    def lbasis(self, X, i):
        x, y = X
        nil = np.zeros_like(x)
        if i == 0:
            phi = np.array([y - 1.0, nil])
        elif i == 1:
            phi = np.array([nil, x])
        elif i == 2:
            phi = np.array([y, nil])
        elif i == 3:
            phi = np.array([nil, x - 1.0])
        else:
            self._index_error()
        dphi = np.ones_like(x) * (-1) ** (i + 1)
        return phi, dphi
