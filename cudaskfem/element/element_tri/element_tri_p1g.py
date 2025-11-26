import numpy as np

from . import ElementTriMorley


class ElementTriP1G(ElementTriMorley):
    """Linear element using :class:`~skfem.element.ElementGlobal`."""

    nodal_dofs = 1
    facet_dofs = 0
    maxdeg = 1
    dofnames = ['u']
    doflocs = np.array([[0., 0.],
                        [1., 0.],
                        [0., 1.]])

    def gdof(self, F, w, i):
        if i == 0:
            return F[()](*w['v'][0])
        elif i == 1:
            return F[()](*w['v'][1])
        elif i == 2:
            return F[()](*w['v'][2])
        self._index_error()
