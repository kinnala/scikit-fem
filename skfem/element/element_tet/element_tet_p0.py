import numpy as np
from ..element_h1 import ElementH1
from ...mesh.mesh3d import MeshTet


class ElementTetP0(ElementH1):

    interior_dofs = 1
    dim = 3
    maxdeg = 0
    dofnames = ['u']
    doflocs = np.array([[.5, .5, .5]])
    mesh_type = MeshTet

    def lbasis(self, X, i):
        return 1. + 0. * X[0], 0. * X
