import numpy as np

from .element import Element


class ElementComposite(Element):
    """Combine multiple elements.

    Allows having different basis functions for different components of a
    vectorial solution.

    """

    def __init__(self, *elems):
        self.elems = elems
        self.nodal_dofs = sum([e.nodal_dofs for e in self.elems])
        self.edge_dofs = sum([e.edge_dofs for e in self.elems])
        self.facet_dofs = sum([e.facet_dofs for e in self.elems])
        self.interior_dofs = sum([e.interior_dofs for e in self.elems])
        self.maxdeg = sum([e.maxdeg for e in self.elems])
        self.dofnames = [i + "^" + str(j + 1)
                         for j in range(len(self.elems))
                         for i in self.elems[j].dofnames]

    def _deduce_bfun(self, mapping, i):
        counts = sum([e._bfun_counts(mapping) for e in self.elems])
        ns = []
        if counts[0] > 0:
            tmp = sum([[j] * self.elems[j].nodal_dofs
                       for j in range(len(self.elems))], [])
            ns += sum([tmp for j in range(int(counts[0] / len(tmp)))], [])
        if counts[1] > 0:
            tmp = sum([[j] * self.elems[j].edge_dofs
                   for j in range(len(self.elems))], [])
            ns += sum([tmp for j in range(int(counts[1] / len(tmp)))], [])
        if counts[2] > 0:
            tmp = sum([[j] * self.elems[j].facet_dofs
                   for j in range(len(self.elems))], [])
            ns += sum([tmp for j in range(int(counts[2] / len(tmp)))], [])
        if counts[3] > 0:
            tmp = sum([[j] * self.elems[j].interior_dofs
                   for j in range(len(self.elems))], [])
            ns += sum([tmp for j in range(int(counts[3] / len(tmp)))], [])

        mask = np.array(ns)
        inds = mask.copy()
        for j in range(len(self.elems)):
            maskj = mask == j
            total = np.sum(maskj)
            seq = np.arange(total, dtype=np.int)
            inds[maskj] = seq

        return ns[i], inds[i]

    def gbasis(self, mapping, X, i, **kwargs):
        n, ind = self._deduce_bfun(mapping, i)
        output = []
        for k, e in enumerate(self.elems):
            if n == k:
                output.append(e.gbasis(mapping, X, ind, **kwargs)[0])
            else:
                output.append(e.gbasis(mapping, X, 0, **kwargs)[0]
                              .zeros_like())
        return tuple(output)
