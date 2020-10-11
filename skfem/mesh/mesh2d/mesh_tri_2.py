import warnings

import numpy as np

from .mesh_tri import MeshTri


class MeshTri2(MeshTri):

    _mesh = None
    name = "Quadratic triangular"

    def __init__(self, doflocs, t, **kwargs):

        warnings.warn("MeshTri2 is an experimental feature and "
                      "not governed by the semantic versioning. "
                      "Several features of MeshTri are still "
                      "missing.")

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
        from skfem.mapping import MappingAffine
        self._elem = ElementTriP2()
        self._basis = InteriorBasis(self, self._elem, MappingAffine(self))
        self._mesh = MeshTri.from_basis(self._basis)
        if t.shape[0] == 6:
            self._mesh.p = doflocs
            self._mesh.t = t

    def mapping(self):
        from skfem.mapping import MappingIsoparametric
        return MappingIsoparametric(self._mesh, self._elem)

    def refine(self, n=1):
        super(MeshTri2, self).refine(n)
        self.__init__(self.p, self.t)

    @classmethod
    def init_circle(cls, Nrefs=3):
        m = MeshTri.init_circle(Nrefs)
        m = cls(m.p, m.t)
        D = m._basis.get_dofs(m.boundary_facets()).flatten()
        m._mesh.p[:, D] =\
            m._mesh.p[:, D] / np.linalg.norm(m._mesh.p[:, D], axis=0)
        return m
