import numpy as np
from .element import Element
from .discrete_field import DiscreteField


class ElementVector(Element):

    def __init__(self, elem):
        self.elem = elem

        # multiplicate number of dofs
        self.nodal_dofs = self.elem.nodal_dofs * self.dim
        self.facet_dofs = self.elem.facet_dofs * self.dim
        self.interior_dofs = self.elem.interior_dofs * self.dim
        self.edge_dofs = self.elem.edge_dofs * self.dim

        # new names for the vector components
        self.dofnames = [i + "^" + str(j + 1)
                         for i in elem.dofnames
                         for j in range(self.dim)]
        self.maxdeg = elem.maxdeg
        self.refdom = elem.refdom

        if hasattr(elem, 'doflocs'):
            self.doflocs = np.array([
                elem.doflocs[int(np.floor(float(i) / float(self.dim)))]
                for i in range(self.dim * elem.doflocs.shape[0])
            ])

    @property
    def dim(self):
        return self.elem.dim

    def gbasis(self, mapping, X, i, tind=None):
        """Set correct components to zero based on ``i``."""
        ind = int(np.floor(float(i) / float(self.dim)))
        n = i - self.dim * ind
        fields = []
        for field in self.elem.gbasis(mapping, X, ind, tind)[0]:
            if field is None:
                fields.append(None)
            else:
                tmp = np.zeros((self.dim,) + field.shape)
                tmp[n] = field
                fields.append(tmp)
        return (DiscreteField(*fields),)
