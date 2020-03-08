import numpy as np

from ..element_global import ElementGlobal
from ...mesh.mesh2d import MeshTri


class ElementTriArgyris(ElementGlobal):
    nodal_dofs = 6
    facet_dofs = 1
    dim = 2
    maxdeg = 5
    dofnames = ['u', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yy', 'u_n']
    doflocs = np.array([[0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [.5, 0.],
                        [.5, .5],
                        [0., .5]])
    mesh_type = MeshTri

    def gdof(self, U, v, e, n, i):
        if i < 18:
            j = i % 6
            k = int(i / 6)
            if j == 0:
                return U[()](*v[k])
            elif j == 1:
                return U[(0,)](*v[k])
            elif j == 2:
                return U[(1,)](*v[k])
            elif j == 3:
                return U[(0, 0)](*v[k])
            elif j == 4:
                return U[(0, 1)](*v[k])
            elif j == 5:
                return U[(1, 1)](*v[k])
        elif i == 18:
            return U[(0,)](*e[0]) * n[0, 0] + U[(1,)](*e[0]) * n[0, 1]
        elif i == 19:
            return U[(0,)](*e[1]) * n[1, 0] + U[(1,)](*e[1]) * n[1, 1]
        elif i == 20:
            return U[(0,)](*e[2]) * n[2, 0] + U[(1,)](*e[2]) * n[2, 1]
        self._index_error()
