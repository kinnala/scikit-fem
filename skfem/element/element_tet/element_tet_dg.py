from ..element import Element
from ...refdom import RefTet


class ElementTetDG(Element):
    """Turn a tetrahedral finite element discontinuous by cutting the
    connectivity to the neighbouring elements."""

    refdom = RefTet

    def __init__(self, elem):
        # change all dofs to interior dofs
        self.elem = elem
        self.maxdeg = elem.maxdeg
        self.interior_dofs = (4 * elem.nodal_dofs +
                              4 * elem.facet_dofs +
                              6 * elem.edge_dofs +
                              elem.interior_dofs)
        self.dofnames = (
            4 * elem.dofnames[:(elem.nodal_dofs + elem.facet_dofs)] +
            6 * elem.dofnames[(elem.nodal_dofs
                               + elem.facet_dofs):(elem.nodal_dofs
                                                   + elem.facet_dofs
                                                   + elem.edge_dofs)] +
            elem.dofnames[(elem.nodal_dofs
                           + elem.facet_dofs
                           + elem.edge_dofs):]
        )
        self.doflocs = elem.doflocs

    def gbasis(self, *args, **kwargs):
        """Call :meth:`Element.gbasis` from the original element."""
        return self.elem.gbasis(*args, **kwargs)
