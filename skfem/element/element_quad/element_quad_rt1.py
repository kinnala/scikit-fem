import numpy as np

from ..element_hdiv import ElementHdiv
from ...refdom import RefQuad


class ElementQuadRT1(ElementHdiv):
    """The lowest order Raviart-Thomas element."""

    facet_dofs = 1
    maxdeg = 1
    dofnames = ["u^n"]
    doflocs = np.array([[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]])
    refdom = RefQuad

    def lbasis(self, X, i):
        x, y = X
        zero = np.zeros_like(x)
        if i == 0:
            phi = np.array([zero, y - 1.0])
        elif i == 1:
            phi = np.array([x, zero])
        elif i == 2:
            phi = np.array([zero, y])
        elif i == 3:
            phi = np.array([x - 1.0, zero])
        else:
            self._index_error()
        dphi = np.ones_like(x)
        return phi, dphi
