import numpy as np

from ..element_global import ElementGlobal
from ...refdom import RefQuad


class ElementQuadBFS(ElementGlobal):

    nodal_dofs = 4
    maxdeg = 6
    tensorial_basis = True
    dofnames = ['u', 'u_x', 'u_y', 'u_xy']
    doflocs = np.array([[0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 1.],
                        [1., 1.],
                        [1., 1.],
                        [1., 1.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.]])
    refdom = RefQuad

    def gdof(self, F, w, i):
        if i >= 16 or i < 0:
            self._index_error()
        j = i % 4
        k = int(i / 4)
        if j == 0:
            return F[()](*w['v'][k])
        elif j == 1:
            return F[(0,)](*w['v'][k])
        elif j == 2:
            return F[(1,)](*w['v'][k])
        elif j == 3:
            return F[(0, 1)](*w['v'][k])
        self._index_error()
