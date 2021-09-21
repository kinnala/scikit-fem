import numpy as np

from ..element_global import ElementGlobal
from ...refdom import RefTri


class ElementTri15ParamPlate(ElementGlobal):
    """15-parameter nonconforming plate element."""

    nodal_dofs = 3
    facet_dofs = 2
    maxdeg = 4
    dofnames = ['u', 'u_x', 'u_y', 'u', 'u_n']
    doflocs = np.array([[0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [.5, 0.],
                        [.5, .5],
                        [0., .5],
                        [.5, 0.],
                        [.5, .5],
                        [0., .5]])

    refdom = RefTri

    def gdof(self, F, w, i):
        if i == 0:
            return F[()](*w['v'][0])
        elif i == 1:
            return F[(0,)](*w['v'][0])
        elif i == 2:
            return F[(1,)](*w['v'][0])
        elif i == 3:
            return F[()](*w['v'][1])
        elif i == 4:
            return F[(0,)](*w['v'][1])
        elif i == 5:
            return F[(1,)](*w['v'][1])
        elif i == 6:
            return F[()](*w['v'][2])
        elif i == 7:
            return F[(0,)](*w['v'][2])
        elif i == 8:
            return F[(1,)](*w['v'][2])
        elif i == 9:
            return F[()](*w['e'][0])
        elif i == 10:
            return (F[(0,)](*w['e'][0]) * w['n'][0, 0]
                    + F[(1,)](*w['e'][0]) * w['n'][0, 1])
        elif i == 11:
            return F[()](*w['e'][1])
        elif i == 12:
            return (F[(0,)](*w['e'][1]) * w['n'][1, 0]
                    + F[(1,)](*w['e'][1]) * w['n'][1, 1])
        elif i == 13:
            return F[()](*w['e'][2])
        elif i == 14:
            return (F[(0,)](*w['e'][2]) * w['n'][2, 0]
                    + F[(1,)](*w['e'][2]) * w['n'][2, 1])
        self._index_error()
