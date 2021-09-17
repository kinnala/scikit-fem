import numpy as np

from .element_h1 import ElementH1
from ..refdom import RefWedge


class ElementWedge1(ElementH1):

    nodal_dofs = 1
    maxdeg = 2
    dofnames = ['u']
    doflocs = np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [1., 0., 1.],
                        [0., 1., 1.]])
    refdom = RefWedge

    def lbasis(self, X, i):
        x, y, z = X

        if i == 0:
            phi = (1. - x - y) * (1. - z)
            dphi = np.array([
                z - 1.,
                z - 1.,
                x + y - 1.,
            ])
        elif i == 1:
            phi = x * (1. - z)
            dphi = np.array([
                1. - z,
                0. * x,
                -x,
            ])
        elif i == 2:
            phi = y * (1. - z)
            dphi = np.array([
                0. * x,
                1. - z,
                -y,
            ])
        elif i == 3:
            phi = (1. - x - y) * z
            dphi = np.array([
                -z,
                -z,
                1. - x - y,
            ])
        elif i == 4:
            phi = x * z
            dphi = np.array([
                z,
                0. * x,
                x,
            ])
        elif i == 5:
            phi = y * z
            dphi = np.array([
                0. * x,
                z,
                y,
            ])
        else:
            self._index_error()

        return phi, dphi
