from ..element import Element


class ElementDG(Element):
    """Turn a finite element discontinuous by cutting connectivity."""

    def __init__(self, elem):

        self.elem = elem
        self.maxdeg = elem.maxdeg
        self.interior_dofs = (elem.refdom.nnodes * elem.nodal_dofs
                              + elem.refdom.nfacets * elem.facet_dofs
                              + elem.refdom.nedges * elem.edge_dofs
                              + elem.interior_dofs)
        # repeat the names of each entity type in the order of the local
        # basis functions: nodal, edge, facet, interior
        n0 = elem.nodal_dofs
        n1 = n0 + elem.edge_dofs
        n2 = n1 + elem.facet_dofs
        self.dofnames = (
            elem.refdom.nnodes * list(elem.dofnames[:n0])
            + elem.refdom.nedges * list(elem.dofnames[n0:n1])
            + elem.refdom.nfacets * list(elem.dofnames[n1:n2])
            + list(elem.dofnames[n2:])
        )
        self.doflocs = elem.doflocs
        self.refdom = elem.refdom

    def gbasis(self, *args, **kwargs):
        """Call :meth:`Element.gbasis` from the original element."""
        return self.elem.gbasis(*args, **kwargs)

    def lbasis(self, *args, **kwargs):
        """Call :meth:`Element.lbasis` from the original element."""
        return self.elem.lbasis(*args, **kwargs)
