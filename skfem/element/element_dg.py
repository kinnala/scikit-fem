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
        self.dofnames = (
            elem.refdom.nnodes * elem.dofnames[:elem.nodal_dofs]
            + elem.refdom.nfacets * elem.dofnames[slice(elem.nodal_dofs,
                                                        (elem.nodal_dofs
                                                         + elem.facet_dofs))]
            + elem.refdom.nedges * elem.dofnames[slice((elem.nodal_dofs
                                                        + elem.facet_dofs),
                                                       (elem.nodal_dofs
                                                        + elem.facet_dofs
                                                        + elem.edge_dofs))]
            + elem.dofnames[(elem.nodal_dofs
                             + elem.facet_dofs
                             + elem.edge_dofs):]
        )
        self.doflocs = elem.doflocs
        self.refdom = elem.refdom

    def gbasis(self, *args, **kwargs):
        """Call :meth:`Element.gbasis` from the original element."""
        return self.elem.gbasis(*args, **kwargs)

    def lbasis(self, *args, **kwargs):
        """Call :meth:`Element.lbasis` from the original element."""
        return self.elem.lbasis(*args, **kwargs)
