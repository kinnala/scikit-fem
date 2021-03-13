from ..element_h1 import ElementH1
from ...refdom import RefQuad


class ElementQuadDG(ElementH1):

    refdom = RefQuad

    def __init__(self, elem):
        # change all dofs to interior dofs
        self.elem = elem
        self.maxdeg = elem.maxdeg
        self.interior_dofs = (4 * elem.nodal_dofs +
                              4 * elem.facet_dofs +
                              elem.interior_dofs)
        self.dofnames = elem.dofnames
        self.doflocs = elem.doflocs

    def lbasis(self, X, i):
        return self.elem.lbasis(X, i)
