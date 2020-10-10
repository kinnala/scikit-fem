import numpy as np

from .mesh_tri import MeshTri


class MeshTri2(MeshTri):

    _mesh = None
    name = "Quadratic triangular"

    def _fix_order(self, doflocs, t):
        """Perform remapping in case doflocs are not given in the same order
        as the DOFs in :class:`skfem.assembly.Dofs` object."""

        # first set
        dofs = t[:3]
        remap = np.zeros(np.max(dofs) + 1, dtype=np.int)
        udofs = np.unique(dofs)
        remap[udofs] = np.arange(len(udofs), dtype=np.int)
        dofs = remap[dofs]

        # second set
        dofs2 = t[3:]
        remap2 = np.zeros(np.max(dofs2) + 1, dtype=np.int)
        udofs2 = np.unique(dofs2)
        remap2[udofs2] = np.arange(len(udofs2), dtype=np.int) + np.max(dofs) + 1
        dofs2 = remap2[dofs2]
        
        doflocs = np.hstack((doflocs[:, udofs], doflocs[:, udofs2]))
        return doflocs, np.vstack((dofs, dofs2))

    def __init__(self, doflocs, t, **kwargs):

        if t.shape[0] == 6:
            doflocs, t = self._fix_order(doflocs, t)
            super(MeshTri2, self).__init__(
                doflocs[:, :(np.max(t[:3]) + 1)],
                t[:3],
                sort_t=False,
                **kwargs
            )
            from skfem.element import ElementTriP2
            from skfem.assembly import InteriorBasis
            self._elem = ElementTriP2()
            self._basis = InteriorBasis(self, self._elem)
            self._mesh = MeshTri.from_basis(self._basis)
            self._mesh.p = doflocs
            self._mesh.t = t
        else:
            # fallback for refinterp
            super(MeshTri2, self).__init__(doflocs, t, **kwargs)

    def mapping(self):
        if self._mesh is not None:
            from skfem.mapping import MappingIsoparametric
            return MappingIsoparametric(self._mesh, self._elem)
        return super(MeshTri2, self).mapping()
