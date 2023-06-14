import numpy as np

from ..element_global import ElementGlobal
from ...refdom import RefHex


class ElementHexC1(ElementGlobal):
    """C1-continuous hex element.

    From: https://doi.org/10.1002/nme.2449

    """
    nodal_dofs = 8
    maxdeg = 6  # todo: unsync integration order and basis degree
    tensorial_basis = True
    dofnames = ['u', 'u_x', 'u_y', 'u_z', 'u_xy', 'u_xz', 'u_yz', 'u_xyz']
    doflocs = np.array([[1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 0.],
                        [1., 1., 0.],
                        [1., 1., 0.],
                        [1., 1., 0.],
                        [1., 1., 0.],
                        [1., 1., 0.],
                        [1., 1., 0.],
                        [1., 1., 0.],
                        [1., 0., 1.],
                        [1., 0., 1.],
                        [1., 0., 1.],
                        [1., 0., 1.],
                        [1., 0., 1.],
                        [1., 0., 1.],
                        [1., 0., 1.],
                        [1., 0., 1.],
                        [0., 1., 1.],
                        [0., 1., 1.],
                        [0., 1., 1.],
                        [0., 1., 1.],
                        [0., 1., 1.],
                        [0., 1., 1.],
                        [0., 1., 1.],
                        [0., 1., 1.],
                        [1., 0., 0.],
                        [1., 0., 0.],
                        [1., 0., 0.],
                        [1., 0., 0.],
                        [1., 0., 0.],
                        [1., 0., 0.],
                        [1., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 1., 0.],
                        [0., 1., 0.],
                        [0., 1., 0.],
                        [0., 1., 0.],
                        [0., 1., 0.],
                        [0., 1., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [0., 0., 1.],
                        [0., 0., 1.],
                        [0., 0., 1.],
                        [0., 0., 1.],
                        [0., 0., 1.],
                        [0., 0., 1.],
                        [0., 0., 1.],
                        [0., 0., 0.],
                        [0., 0., 0.],
                        [0., 0., 0.],
                        [0., 0., 0.],
                        [0., 0., 0.],
                        [0., 0., 0.],
                        [0., 0., 0.],
                        [0., 0., 0.]])
    refdom = RefHex
    derivatives = 3

    def gdof(self, F, w, i):
        if i >= 64 or i < 0:
            self._index_error()
        j = i % 8
        k = int(i / 8)
        if j == 0:
            return F[()](*w['v'][k])
        elif j == 1:
            return F[(0,)](*w['v'][k])
        elif j == 2:
            return F[(1,)](*w['v'][k])
        elif j == 3:
            return F[(2,)](*w['v'][k])
        elif j == 4:
            return F[(0, 1)](*w['v'][k])
        elif j == 5:
            return F[(0, 2)](*w['v'][k])
        elif j == 6:
            return F[(1, 2)](*w['v'][k])
        elif j == 7:
            return F[(0, 1, 2)](*w['v'][k])
        self._index_error()
