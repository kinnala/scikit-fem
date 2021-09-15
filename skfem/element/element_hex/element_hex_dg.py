from ..element_h1 import ElementH1
from ...refdom import RefHex


class ElementHexDG(ElementH1):

    refdom = RefHex

    def __init__(self, elem):
        # change all dofs to interior dofs
        self.elem = elem
        self.maxdeg = elem.maxdeg
        self.interior_dofs = (8 * elem.nodal_dofs +
                              6 * elem.facet_dofs +
                              12 * elem.edge_dofs +
                              elem.interior_dofs)
        self.dofnames = (
            8 * elem.dofnames[:elem.nodal_dofs] +
            6 * elem.dofnames[elem.nodal_dofs:(elem.nodal_dofs
                                               + elem.facet_dofs)] +
            12 * elem.dofnames[(elem.nodal_dofs
                                + elem.facet_dofs):(elem.nodal_dofs
                                                    + elem.facet_dofs
                                                    + elem.edge_dofs)] +
            elem.dofnames[(elem.nodal_dofs
                           + elem.facet_dofs
                           + elem.edge_dofs):]
        )
        self.doflocs = elem.doflocs

    def lbasis(self, X, i):
        return self.elem.lbasis(X, i)
