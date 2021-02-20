import numpy as np

from ..element_global import ElementGlobal
from ...refdom import RefTri


class ElementTriArgyris(ElementGlobal):

    nodal_dofs = 6
    facet_dofs = 1
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
    refdom = RefTri

    def gdof(self, F, w, i):
        if i < 18:
            j = i % 6
            k = int(i / 6)
            if j == 0:
                return F[()](*w['v'][k])
            elif j == 1:
                return F[(0,)](*w['v'][k])
            elif j == 2:
                return F[(1,)](*w['v'][k])
            elif j == 3:
                return F[(0, 0)](*w['v'][k])
            elif j == 4:
                return F[(0, 1)](*w['v'][k])
            elif j == 5:
                return F[(1, 1)](*w['v'][k])
        elif i == 18:
            return (F[(0,)](*w['e'][0]) * w['n'][0, 0] +
                    F[(1,)](*w['e'][0]) * w['n'][0, 1])
        elif i == 19:
            return (F[(0,)](*w['e'][1]) * w['n'][1, 0] +
                    F[(1,)](*w['e'][1]) * w['n'][1, 1])
        elif i == 20:
            return (F[(0,)](*w['e'][2]) * w['n'][2, 0] +
                    F[(1,)](*w['e'][2]) * w['n'][2, 1])
        self._index_error()
