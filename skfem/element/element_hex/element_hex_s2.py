import numpy as np

from ..element_h1 import ElementH1
from ...mesh.mesh3d import MeshHex


class ElementHexS2(ElementH1):
    nodal_dofs = 1
    edge_dofs = 1
    dim = 3
    maxdeg = 4
    dofnames = ['u', 'u']
    doflocs = np.array([[1., 1., 1.],
                        [1., 1., 0.], # 1
                        [1., 0., 1.],
                        [0., 1., 1.],
                        [1., 0., 0.], # 4
                        [0., 1., 0.],
                        [0., 0., 1.], # 6
                        [0., 0., 0.],
                        [1., 1., .5], # 0->1
                        [1., .5, 1.], # 0->2
                        [.5, 1., 1.], # 0->3
                        [1., .5, 0.], # 1->4
                        [.5, 1., 0.], # 1->5
                        [1., 0., .5], # 2->4
                        [.5, 0., 1.], # 2->6
                        [0., 1., .5], # 3->5
                        [0., .5, 1.], # 3->6
                        [.5, 0., 0.], # 4->7
                        [0., .5, 0.],
                        [0., 0., .5],
                        ])
    mesh_type = MeshHex

    def lbasis(self, X, i):
        x, y, z = 2 * X - 1

        if i == 0:
            phi = (1 + x) * (1 + y) * (1 + z) * (x + y + z - 2) / 8
            dphi = np.array([(1 + y) * (1 + z) * (x + y + z - 2) / 8\
                             + (1 + x) * (1 + y) * (1 + z) / 8,
                             (1 + x) * (1 + z) * (x + y + z - 2) / 8\
                             + (1 + x) * (1 + y) * (1 + z) / 8,
                             (1 + x) * (1 + y) * (x + y + z - 2) / 8\
                             + (1 + x) * (1 + y) * (1 + z) / 8])
        elif i == 1:
            phi = (1 + x) * (1 + y) * (1 - z) * (x + y - z - 2) / 8
            dphi = np.array([(1 + y) * (1 - z) * (x + y - z - 2) / 8\
                             + (1 + x) * (1 + y) * (1 - z) / 8,
                             (1 + x) * (1 - z) * (x + y - z - 2) / 8\
                             + (1 + x) * (1 + y) * (1 - z) / 8,
                             - (1 + x) * (1 + y) * (x + y - z - 2) / 8\
                             - (1 + x) * (1 + y) * (1 - z) / 8])
        elif i == 2:
            phi = (1 + x) * (1 - y) * (1 + z) * (x - y + z - 2) / 8
            dphi = np.array([(1 - y) * z,
                             -x * z,
                             x * (1 - y)])
        elif i == 3:
            phi = (1 - x) * (1 + y) * (1 + z) * (- x + y + z - 2) / 8
            dphi = np.array([-y * z,
                             (1 - x) * z,
                             (1 - x) * y])
        elif i == 4:
            phi = (1 + x) * (1 - y) * (1 - z) * (x - y - z - 2) / 8
            dphi = np.array([(1 - y) * (1 - z),
                             -x * (1 - z),
                             -x * (1 - y)])
        elif i == 5:
            phi = (1 - x) * (1 + y) * (1 - z) * (- x + y - z - 2) / 8
            dphi = np.array([-y * (1 - z),
                             (1 - x) * (1 - z),
                             -(1 - x) * y])
        elif i == 6:
            phi = (1 - x) * (1 - y) * (1 + z) * (- x - y + z - 2) / 8
            dphi = np.array([-(1 - y) * z,
                             -(1 - x) * z,
                             (1 - x) * (1 - y)])
        elif i == 7:
            phi = (1 - x) * (1 - y) * (1 - z) * (- x - y - z - 2) / 8
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        # phi ok
        elif i == 8:
            phi = (1 + x) * (1 + y) * (1 - z ** 2) / 4
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 9:
            phi = (1 + x) * (1 - y ** 2) * (1 + z) / 4
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 10:
            phi = (1 - x ** 2) * (1 + y) * (1 + z) / 4 # ok
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 11:
            phi = (1 + x) * (1 - y ** 2) * (1 - z) / 4 # ok
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 12:
            phi = (1 - x ** 2) * (1 + y) * (1 - z) / 4 # ok
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 13:
            phi = (1 + x) * (1 - y) * (1 - z ** 2) / 4 # ok
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 14:
            phi = (1 - x ** 2) * (1 - y) * (1 + z) / 4 # ok
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 15:
            phi = (1 - x) * (1 + y) * (1 - z ** 2) / 4 # ok
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 16:
            phi = (1 - x) * (1 - y ** 2) * (1 + z) / 4 # ok
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 17:
            phi = (1 - x ** 2) * (1 - y) * (1 - z) / 4
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 18:
            phi = (1 - x) * (1 - y ** 2) * (1 - z) / 4
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        elif i == 19:
            phi = (1 - x) * (1 - y) * (1 - z ** 2) / 4
            dphi = np.array([-(1 - y) * (1 - z),
                             -(1 - x) * (1 - z),
                             -(1 - x) * (1 - y)])
        else:
            self._index_error()

        return phi, dphi
