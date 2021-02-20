import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefQuad


class ElementQuad1(ElementH1):

    nodal_dofs = 1
    maxdeg = 2
    dofnames = ['u']
    doflocs = np.array([[0., 0.],
                        [1., 0.],
                        [1., 1.],
                        [0., 1.]])
    refdom = RefQuad

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = (1. - x) * (1. - y)
            dphi = np.array([-1. + y, -1. + x])
        elif i == 1:
            phi = x * (1. - y)
            dphi = np.array([1. - y, -x])
        elif i == 2:
            phi = x * y
            dphi = np.array([y, x])
        elif i == 3:
            phi = (1. - x) * y
            dphi = np.array([-y, 1. - x])
        else:
            self._index_error()

        return phi, dphi
