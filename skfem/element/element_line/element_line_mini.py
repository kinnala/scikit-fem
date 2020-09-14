import numpy as np

from ..element_h1 import ElementH1
from ...mesh.mesh_line import MeshLine


class ElementLineMini(ElementH1):

    nodal_dofs = 1
    interior_dofs = 1
    dim = 1
    maxdeg = 1 + dim
    dofnames = ['u', 'NA']
    doflocs = np.array([[0.],
                        [1.],
                        [np.nan] * dim])
    mesh_type = MeshLine

    def lbasis(self, X, i):
        x = X[0]

        if i == 0:
            phi = 1. - x
            dphi = np.array([-1. + 0. * x])
        elif i == 1:
            phi = x
            dphi = np.array([1. + 0. * x])
        elif i == 2:
            phi = 4 * x * (1. - x)
            dphi = np.array([4. * (1. - 2. * x)])
        else:
            self._index_error()

        return phi, dphi
