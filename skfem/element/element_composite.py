from .element import Element


class ElementComposite(Element):

    def __init__(self, *elems):
        self.elems = elems
        self.nodal_dofs = sum([e.nodal_dofs for e in self.elems])
        self.edge_dofs = sum([e.edge_dofs for e in self.elems])
        self.facet_dofs = sum([e.facet_dofs for e in self.elems])
        self.interior_dofs = sum([e.interior_dofs for e in self.elems])
        self.maxdeg = sum([e.maxdeg for e in self.elems])

    def _deduce_bfun(self, mapping, i):
        e1 = self.elems[0]._bfun_counts(mapping)
        e2 = self.elems[1]._bfun_counts(mapping)
        if i < e1[0]:
            return 0, i
        elif i < e1[0] + e2[0]:
            return 1, i - e1[0]
        elif i < e1[0] + e2[0] + e1[1]:
            return 0, i - e2[0]
        elif i < e1[0] + e2[0] + e1[1] + e2[1]:
            return 1, i - e1[0] - e1[1]
        elif i < e1[0] + e2[0] + e1[1] + e2[1] + e1[2]:
            return 0, i - e2[0] - e2[1]
        elif i < e1[0] + e2[0] + e1[1] + e2[1] + e1[2] + e2[2]:
            return 1, i - e1[0] - e1[1] - e1[2]
        elif i < e1[0] + e2[0] + e1[1] + e2[1] + e1[2] + e2[2] + e1[3]:
            return 0, i - e2[0] - e2[1] - e2[2]
        return 1, i - e1[0] - e1[1] - e1[2] - e1[3]

    def gbasis(self, mapping, X, i, **kwargs):
        n, ind = self._deduce_bfun(mapping, i)
        field = self.elems[n].gbasis(mapping, X, ind, **kwargs)[0]
        if n == 0:
            Z = self.elems[1].gbasis(mapping, X, 0, **kwargs)[0].zeros_like()
            return (field, Z)
        elif n == 1:
            Z = self.elems[0].gbasis(mapping, X, 0, **kwargs)[0].zeros_like()
            return (Z, field)
