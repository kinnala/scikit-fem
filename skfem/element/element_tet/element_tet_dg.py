from ..element import Element
from ...refdom import RefTet


class ElementTetDG(Element):

    refdom = RefTet

    def __init__(self, elem):
        # change all dofs to interior dofs
        self.elem = elem
        self.maxdeg = elem.maxdeg
        self.interior_dofs = (4 * elem.nodal_dofs +
                              4 * elem.facet_dofs +
                              6 * elem.edge_dofs +
                              elem.interior_dofs)
        self.dofnames = elem.dofnames
        self.doflocs = elem.doflocs

    def gbasis(self, *args, **kwargs):
        return self.elem.gbasis(*args, **kwargs)
