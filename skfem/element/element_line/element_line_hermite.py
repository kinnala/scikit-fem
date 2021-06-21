import numpy as np

from ..element_global import ElementGlobal
from ...refdom import RefLine


class ElementLineHermite(ElementGlobal):
    """:math:`H^2`-conforming element with 4 DOFs."""

    nodal_dofs = 2
    maxdeg = 3
    dofnames = ['u', 'u_x']
    doflocs = np.array([[0.],
                        [0.],
                        [1.],
                        [1.]])
    refdom = RefLine

    def gdof(self, F, w, i):
        if i == 0:
            return F[()](*w['v'][0])
        elif i == 1:
            return F[(0,)](*w['v'][0])
        elif i == 2:
            return F[()](*w['v'][1])
        elif i == 3:
            return F[(0,)](*w['v'][1])
        self._index_error()
