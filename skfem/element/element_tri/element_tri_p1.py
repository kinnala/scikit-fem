import numpy as np

from ..element_h1 import ElementH1
from ..element_line.element_line_p1 import ElementLineP1
from ...mesh.mesh2d import MeshTri


class ElementTriP1(ElementH1):
    nodal_dofs = 1
    dim = 2
    maxdeg = 1
    dofnames = ['u']
    doflocs = np.array([[0., 0.],
                        [1., 0.],
                        [0., 1.]])
    mesh_type = MeshTri
    boundary_element = ElementLineP1()

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
