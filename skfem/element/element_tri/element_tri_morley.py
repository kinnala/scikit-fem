import numpy as np

from ..element_global import ElementGlobal
from ...mesh.mesh2d import MeshTri


class ElementTriMorley(ElementGlobal):
    nodal_dofs = 1
    facet_dofs = 1
    dim = 2
    maxdeg = 2
    dofnames = ['u', 'u_n']
    doflocs = np.array([[0., 0.],
                        [1., 0.],
                        [0., 1.],
                        [.5, 0.],
                        [.5, .5],
                        [0., .5]])
    mesh_type = MeshTri

    def gdof(self, U, v, e, n, i):
        if i == 0:
            return U[()](*v[0])
        elif i == 1:
            return U[()](*v[1])
        elif i == 2:
            return U[()](*v[2])
        elif i == 3:
            return U[(0,)](*e[0]) * n[0, 0] + U[(1,)](*e[0]) * n[0, 1]
        elif i == 4:
            return U[(0,)](*e[1]) * n[1, 0] + U[(1,)](*e[1]) * n[1, 1]
        elif i == 5:
            return U[(0,)](*e[2]) * n[2, 0] + U[(1,)](*e[2]) * n[2, 1]
        self._index_error()
