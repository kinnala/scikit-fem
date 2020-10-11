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
            from skfem.element import ElementQuad2
            from skfem.assembly import InteriorBasis
            self._elem = ElementQuad2()
            self._basis = InteriorBasis(self, self._elem)
            self._mesh = MeshQuad.from_basis(self._basis)
            self._mesh.p = doflocs
            self._mesh.t = t
        else:
            # fallback for refinterp
            super(MeshQuad2, self).__init__(doflocs, t, **kwargs)

    def mapping(self):
        if self._mesh is not None:
            from skfem.mapping import MappingIsoparametric
            return MappingIsoparametric(self._mesh, self._elem)
        return super(MeshQuad2, self).mapping()
