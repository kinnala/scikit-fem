from ..element import Element
from ...refdom import RefTri


class ElementTriDG(Element):
    """Turn a triangular finite element discontinuous by cutting the
    connectivity to the neighbouring elements."""

    refdom = RefTri

    def __init__(self, elem):
        # change all dofs to interior dofs
        self.elem = elem
        self.maxdeg = elem.maxdeg
        self.interior_dofs = (3 * elem.nodal_dofs +
                              3 * elem.facet_dofs +
                              elem.interior_dofs)
        self.dofnames = elem.dofnames
        self.doflocs = elem.doflocs

    def gbasis(self, *args, **kwargs):
        """Call :meth:`Element.gbasis` from the original element."""
        return self.elem.gbasis(*args, **kwargs)
