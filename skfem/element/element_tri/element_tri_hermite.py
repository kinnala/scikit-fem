import numpy as np

from ..element_global import ElementGlobal
from ...refdom import RefTri


class ElementTriHermite(ElementGlobal):

    nodal_dofs = 3
    interior_dofs = 1
    maxdeg = 3
    dofnames = ['u', 'u_x', 'u_y', 'u']
    doflocs = np.array([[0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [1 / 3, 1 / 3]])
    refdom = RefTri

    def gdof(self, F, w, i):
        if i < 9:
            j = i % 3
            k = int(i / 3)
            if j == 0:
                return F[()](*w['v'][k])
            elif j == 1:
                return F[(0,)](*w['v'][k])
            elif j == 2:
                return F[(1,)](*w['v'][k])
        elif i == 9:
            mx = (w['v'][0][0] + w['v'][1][0] + w['v'][2][0]) / 3
            my = (w['v'][0][1] + w['v'][1][1] + w['v'][2][1]) / 3
            return F[()](mx, my)
        self._index_error()
