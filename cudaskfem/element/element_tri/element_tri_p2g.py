import numpy as np

from . import ElementTriMorley


class ElementTriP2G(ElementTriMorley):
    """Quadratic element using :class:`~skfem.element.ElementGlobal`."""

    nodal_dofs = 1
    facet_dofs = 1
    maxdeg = 2
    dofnames = ['u', 'u']
    doflocs = np.array([[0., 0.],
                        [1., 0.],
                        [0., 1.],
                        [.5, 0.],
                        [.5, .5],
                        [0., .5]])

    def gdof(self, F, w, i):
        if i == 0:
            return F[()](*w['v'][0])
        elif i == 1:
            return F[()](*w['v'][1])
        elif i == 2:
            return F[()](*w['v'][2])
        elif i == 3:
            return F[()](*w['e'][0])
        elif i == 4:
            return F[()](*w['e'][1])
        elif i == 5:
            return F[()](*w['e'][2])
        self._index_error()
