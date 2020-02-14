import numpy as np

from ..element_h1 import ElementH1
from ...mesh.mesh3d import MeshHex


class ElementHex1(ElementH1):
    nodal_dofs = 1
    dim = 3
    maxdeg = 3
    dofnames = ['u']
    doflocs = np.array([[-1., -1., -1.],
                        [-1., -1.,  1.],
                        [-1.,  1., -1.],
                        [ 1., -1., -1.],
                        [-1.,  1.,  1.],
                        [ 1., -1.,  1.],
                        [ 1.,  1., -1.],
                        [ 1.,  1.,  1.]])
    mesh_type = MeshHex

    def lbasis(self, X, i):
        x, y, z = X

        if i == 0:
            phi = (1 + x)*(1 + y)*(1 + z)
            dphi = np.array([(1 + y)*(1 + z),
                             (1 + x)*(1 + z),
                             (1 + x)*(1 + y)])
        elif i == 1:
            phi = (1 + x)*(1 + y)*(1 - z)
            dphi = np.array([(1 + y)*(1 - z),
                             (1 + x)*(1 - z),
                             -(1 + x)*(1 + y)])
        elif i == 2:
            phi = (1 + x)*(1 - y)*(1 + z)
            dphi = np.array([(1 - y)*(1 + z),
                             -(1 + x)*(1 + z),
                             (1 + x)*(1 - y)])
        elif i == 3:
            phi = (1 - x)*(1 + y)*(1 + z)
            dphi = np.array([-(1 + y)*(1 + z),
                             (1 - x)*(1 + z),
                             (1 - x)*(1 + y)])
        elif i == 4:
            phi = (1 + x)*(1 - y)*(1 - z)
            dphi = np.array([(1 - y)*(1 - z),
                             -(1 + x)*(1 - z),
                             -(1 + x)*(1 - y)])
        elif i == 5:
            phi = (1 - x)*(1 + y)*(1 - z)
            dphi = np.array([-(1 + y)*(1 - z),
                             (1 - x)*(1 - z),
                             -(1 - x)*(1 + y)])
        elif i == 6:
            phi = (1 - x)*(1 - y)*(1 + z)
            dphi = np.array([-(1 - y)*(1 + z),
                             -(1 - x)*(1 + z),
                             (1 - x)*(1 - y)])
        elif i == 7:
            phi = (1 - x)*(1 - y)*(1 - z)
            dphi = np.array([-(1 - y)*(1 - z),
                             -(1 - x)*(1 - z),
                             -(1 - x)*(1 - y)])
        else:
            self._index_error()

        return 0.125*phi, 0.125*dphi
