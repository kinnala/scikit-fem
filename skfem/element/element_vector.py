import numpy as np
from .element import Element
from .block_array import BlockArray, BlockDiscreteField


class ElementVector(Element):
    """Use the same element for each dimension."""

    def __init__(self, elem, dim=None):
        self.elem = elem
        self._dim = elem.dim if dim is None else dim

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
        return self._dim

    def gbasis(self, mapping, X, i, tind=None):
        """Place the scalar basis function to the correct component.

        Only the nonzero component is stored, see
        :class:`~skfem.element.block_array.BlockArray`.

        """
        ind, n = divmod(int(i), self.dim)
        fields = []
        for field in self.elem.gbasis(mapping, X, ind, tind)[0].astuple:
            if field is None:
                fields.append(None)
            elif isinstance(field, BlockArray):  # nested ElementVector
                fields.append(BlockArray((self.dim,) + field.shape,
                                         field.r + 1,
                                         {(n,) + k: b
                                          for k, b in field.blocks.items()},
                                         field.dtype))
            else:
                field = np.asarray(field)
                fields.append(BlockArray((self.dim,) + field.shape,
                                         1,
                                         {(n,): field},
                                         field.dtype))
        return (BlockDiscreteField(*fields),)
