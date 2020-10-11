import numpy as np

from .mesh_quad import MeshQuad


class MeshQuad2(MeshQuad):

    _mesh = None
    name = "Quadratic quadrilateral"

    def __init__(self, doflocs, t, **kwargs):

        if t.shape[0] == 9:
            dofs, ix = np.unique(t[:4], return_inverse=True)
            super(MeshQuad2, self).__init__(
                doflocs[:, dofs],
                np.arange(len(dofs), dtype=np.int)[ix].reshape(t[:4].shape),
                **kwargs
            )
        else:
            # fallback for refinterp
            super(MeshQuad2, self).__init__(doflocs, t, **kwargs)
        from skfem.element import ElementQuad1, ElementQuad2
        from skfem.assembly import InteriorBasis
        from skfem.mapping import MappingIsoparametric
        self._elem = ElementQuad2()
        self._basis = InteriorBasis(
            self,
            self._elem,
            MappingIsoparametric(self, ElementQuad1())
        )
        self._mesh = MeshQuad.from_basis(self._basis)
        if t.shape[0] == 9:
            self._mesh.p = doflocs
            self._mesh.t = t

    def refine(self, n=1):
        super(MeshQuad2, self).refine(n)
        self.__init__(self.p, self.t)

    def mapping(self):
        from skfem.mapping import MappingIsoparametric
        return MappingIsoparametric(self._mesh, self._elem)
