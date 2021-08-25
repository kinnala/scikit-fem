from dataclasses import replace

import numpy as np


class MeshDG:

    @classmethod
    def periodic(cls, mesh, ix, ix0):
        """Initialize a periodic mesh from a standard (nonperiodic) mesh.

        The mesh types that can describe a discontinuous topology include:

        - :class:`~skfem.mesh.MeshTri1DG`
        - :class:`~skfem.mesh.MeshQuad1DG`
        - :class:`~skfem.mesh.MeshHex1DG`
        - :class:`~skfem.mesh.MeshLine1DG`

        Parameters
        ----------
        mesh
            The mesh to periodify.
        ix
            An array of nodes to eliminate.  They are replaced by the nodes in
            the array ``ix``.
        ix0
            An array of nodes left unchanged.

        """
        assert cls.elem.interior_dofs > 0
        assert cls.elem.refdom == mesh.elem.refdom
        assert len(ix) == len(ix0)

        # reorder vertices: eliminated nodes must have highest index values
        nix = np.arange(mesh.nvertices - len(ix),
                        mesh.nvertices,
                        dtype=np.int64)
        remap = (-16) * np.ones(mesh.nvertices, dtype=np.int64)
        remap[ix] = nix
        oix = (remap == -16).nonzero()[0]
        remap[oix] = np.arange(mesh.nvertices - len(ix), dtype=np.int64)

        doflocs = np.hstack((mesh.doflocs[:, oix], mesh.doflocs[:, ix]))
        t = remap[mesh.t]

        reordered_mesh = replace(
            mesh,
            doflocs=doflocs,
            t=t,
            sort_t=False,
        )

        # make periodic
        reremap = np.arange(mesh.nvertices, dtype=np.int64)
        ix1 = remap[ix]
        ix2 = remap[ix0]
        reremap[ix1] = ix2

        periodic_mesh = cls.from_mesh(reordered_mesh, reremap[t])

        # store reordered mesh and reverse mapping
        periodic_mesh._unperiodic = reordered_mesh
        periodic_mesh._nodes = reremap

        return periodic_mesh
