import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefHex


class ElementHexS2(ElementH1):

    nodal_dofs = 1
    edge_dofs = 1
    maxdeg = 4
    dofnames = ['u', 'u']
    doflocs = np.array([[1., 1., 1.],
                        [1., 1., 0.],
                        [1., 0., 1.],
                        [0., 1., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [0., 0., 0.],
                        [1., 1., .5],
                        [1., .5, 1.],
                        [.5, 1., 1.],
                        [1., .5, 0.],
                        [.5, 1., 0.],
                        [1., 0., .5],
                        [.5, 0., 1.],
                        [0., 1., .5],
                        [0., .5, 1.],
                        [.5, 0., 0.],
                        [0., .5, 0.],
                        [0., 0., .5]])
    refdom = RefHex

    def lbasis(self, X, i):
        x, y, z = 2 * X - 1

        if i < 8:
            s = [
                (1, 1, 1),
                (1, 1, -1),
                (1, -1, 1),
                (-1, 1, 1),
                (1, -1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (-1, -1, -1),
            ][i]
            x *= s[0]
            y *= s[1]
            z *= s[2]
            phi = (1 + x) * (1 + y) * (1 + z) * (x + y + z - 2) / 8
            dphi = np.array([s[0] * (1 + y) * (1 + z) * (x + y + z - 2)
                             + s[0] * (1 + x) * (1 + y) * (1 + z),
                             s[1] * (1 + x) * (1 + z) * (x + y + z - 2)
                             + s[1] * (1 + x) * (1 + y) * (1 + z),
                             s[2] * (1 + x) * (1 + y) * (x + y + z - 2)
                             + s[2] * (1 + x) * (1 + y) * (1 + z)]) / 8
        elif i < 20:
            s = [
                (1, 1, -z),
                (1, -y, 1),
                (-x, 1, 1),
                (1, -y, -1),
                (-x, 1, -1),
                (1, -1, -z),
                (-x, -1, 1),
                (-1, 1, -z),
                (-1, -y, 1),
                (-x, -1, -1),
                (-1, -y, -1),
                (-1, -1, -z),
            ][i - 8]
            x *= s[0]
            y *= s[1]
            z *= s[2]
            phi = (1 + x) * (1 + y) * (1 + z) / 4
            dx = 2. * s[0] if isinstance(s[0], np.ndarray) else s[0]
            dy = 2. * s[1] if isinstance(s[1], np.ndarray) else s[1]
            dz = 2. * s[2] if isinstance(s[2], np.ndarray) else s[2]
            dphi = np.array([dx * (1 + y) * (1 + z),
                             dy * (1 + x) * (1 + z),
                             dz * (1 + x) * (1 + y)]) / 4
        else:
            self._index_error()

        return phi, 2 * dphi
