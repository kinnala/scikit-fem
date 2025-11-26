import numpy as np

from ..element_global import ElementGlobal
from ...refdom import RefQuad


class ElementQuad2G(ElementGlobal):
    """Global quadratic element for quadrilaterals."""

    nodal_dofs = 1
    facet_dofs = 1
    interior_dofs = 1
    maxdeg = 4
    tensorial_basis = True
    dofnames = ['u', 'u', 'u']
    doflocs = np.array([[0., 0.],
                        [1., 0.],
                        [1., 1.],
                        [0., 1.],
                        [.5, 0.],
                        [1., .5],
                        [.5, 1.],
                        [0., .5],
                        [.5, .5]])
    refdom = RefQuad

    def gdof(self, F, w, i):
        if i == 0:
            return F[()](*w['v'][0])
        elif i == 1:
            return F[()](*w['v'][1])
        elif i == 2:
            return F[()](*w['v'][2])
        elif i == 3:
            return F[()](*w['v'][3])
        elif i == 4:
            return F[()](*w['e'][0])
        elif i == 5:
            return F[()](*w['e'][1])
        elif i == 6:
            return F[()](*w['e'][2])
        elif i == 7:
            return F[()](*w['e'][3])
        elif i == 8:
            return F[()](*(.25 * (w['v'][0]
                                  + w['v'][1]
                                  + w['v'][2]
                                  + w['v'][3])))
        self._index_error()
