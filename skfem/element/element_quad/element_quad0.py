import numpy as np

from ..element_h1 import ElementH1
from ...mesh.mesh2d import MeshQuad


class ElementQuad0(ElementH1):

    interior_dofs = 1
    dim = 2
    maxdeg = 0
    dofnames = ['u']
    doflocs = np.zeros((1, 2))
    mesh_type = MeshQuad

    def lbasis(self, X, i):
        if i == 0:
            return np.ones(X.shape[1:]), np.zeros_like(X)
        else:
            self._index_error()
