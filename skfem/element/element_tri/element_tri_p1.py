import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTri


class ElementTriP1(ElementH1):
    """Piecewise linear element."""

    nodal_dofs = 1
    maxdeg = 1
    dofnames = ['u']
    doflocs = np.array([[0., 0.],
                        [1., 0.],
                        [0., 1.]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = 1. - x - y
            dphi = np.array([-1. + 0. * x, -1. + 0. * x])
        elif i == 1:
            phi = x
            dphi = np.array([1. + 0. * x, 0. * x])
        elif i == 2:
            phi = y
            dphi = np.array([0. * x, 1. + 0. * x])
        else:
            self._index_error()

        return phi, dphi


class ElementTriP1DG(ElementTriP1):

    nodal_dofs = 0
    interior_dofs = 3
    dofnames = ['u'] * 3
