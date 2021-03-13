from typing import List, Any

import numpy as np
from numpy import ndarray

from .element import Element
from .discrete_field import DiscreteField


class ElementComposite(Element):
    """Combine multiple elements.

    Allows using different basis functions for different components of a
    vectorial solution.

    """

    def __init__(self, *elems: Element):
        self.elems = elems
        self.nodal_dofs = sum([e.nodal_dofs for e in self.elems])
        self.edge_dofs = sum([e.edge_dofs for e in self.elems])
        self.facet_dofs = sum([e.facet_dofs for e in self.elems])
        self.interior_dofs = sum([e.interior_dofs for e in self.elems])
        self.maxdeg = sum([e.maxdeg for e in self.elems])

        for e in self.elems:
            if e.refdom is not self.elems[0].refdom:
                raise ValueError("Elements are incompatible.")

        dofnames = []
        for i, e in enumerate(self.elems):  # nodal
            for j in range(e.nodal_dofs):
                dofnames.append(e.dofnames[j] + "^" + str(i + 1))
        for i, e in enumerate(self.elems):  # edge
            for j in range(e.nodal_dofs, e.nodal_dofs + e.edge_dofs):
                dofnames.append(e.dofnames[j] + "^" + str(i + 1))
        for i, e in enumerate(self.elems):  # facet
            for j in range(e.nodal_dofs + e.edge_dofs,
                           e.nodal_dofs + e.edge_dofs + e.facet_dofs):
                dofnames.append(e.dofnames[j] + "^" + str(i + 1))
        for i, e in enumerate(self.elems):  # interior
            for j in range(e.nodal_dofs + e.edge_dofs + e.facet_dofs,
                           (e.nodal_dofs + e.edge_dofs
                            + e.facet_dofs + e.interior_dofs)):
                dofnames.append(e.dofnames[j] + "^" + str(i + 1))
        self.dofnames = dofnames

        doflocs = []
        for i in np.arange(np.sum(np.array([e._bfun_counts()
                                            for e in self.elems])),
                           dtype=int):
            n, ind = self._deduce_bfun(i)
            doflocs.append(self.elems[n].doflocs[ind])
        self.doflocs = np.array(doflocs)

        self.refdom = elems[0].refdom

    @property
    def dim(self):
        return self.elems[0].dim

    def _deduce_bfun(self, i: int):
        """Deduce component and basis function for i'th index."""
        counts = np.sum(np.array([e._bfun_counts()
                                  for e in self.elems]), axis=0)
        tmp: List[Any] = []
        ns: List[Any] = []
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
            seq = np.arange(total, dtype=np.int_)
            inds[maskj] = seq

        return ns[i], inds[i]

    def gbasis(self, mapping, X: ndarray, i: int, tind=None):
        """Call correct :meth:`Element.gbasis` based on ``i``."""
        n, ind = self._deduce_bfun(i)
        output: List[DiscreteField] = []
        for k, e in enumerate(self.elems):
            if n == k:
                output.append(e.gbasis(mapping, X, ind, tind)[0])
            else:
                output.append(e.gbasis(mapping, X, 0, tind)[0].zeros_like())
        return tuple(output)
