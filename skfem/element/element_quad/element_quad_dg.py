from ..element_h1 import ElementH1
from ...refdom import RefQuad


class ElementQuadDG(ElementH1):
    """Turn a quadrilateral finite element discontinuous by cutting the
    connectivity to the neighbouring elements."""

    refdom = RefQuad

    def __init__(self, elem):
        # change all dofs to interior dofs
        self.elem = elem
        self.maxdeg = elem.maxdeg
        self.interior_dofs = (4 * elem.nodal_dofs +
                              4 * elem.facet_dofs +
                              elem.interior_dofs)
        self.dofnames = (
            4 * elem.dofnames[:(elem.nodal_dofs + elem.facet_dofs)] +
            elem.dofnames[(elem.nodal_dofs + elem.facet_dofs):]
        )
        self.doflocs = elem.doflocs

    def lbasis(self, X, i):
        return self.elem.lbasis(X, i)
