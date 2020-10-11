import numpy as np

from .mesh_tri import MeshTri


class MeshTri2(MeshTri):

    _mesh = None
    name = "Quadratic triangular"

    def __init__(self, doflocs, t, **kwargs):

        if t.shape[0] == 6:
            dofs, ix = np.unique(t[:3], return_inverse=True)
            super(MeshTri2, self).__init__(
                doflocs[:, dofs],
                np.arange(len(dofs), dtype=np.int)[ix].reshape(t[:3].shape),
                sort_t=False,
                **kwargs
            )
        else:
            # fallback for refinterp
            super(MeshTri2, self).__init__(doflocs, t, **kwargs)
        from skfem.element import ElementTriP2
        from skfem.assembly import InteriorBasis
        self._elem = ElementTriP2()
        self._basis = InteriorBasis(self, self._elem)
        self._mesh = MeshTri.from_basis(self._basis)
        if t.shape[0] == 6:
            self._mesh.p = doflocs
            self._mesh.t = t

    def mapping(self):
        if self._mesh is not None:
            from skfem.mapping import MappingIsoparametric
            return MappingIsoparametric(self._mesh, self._elem)
        return super(MeshTri2, self).mapping()
