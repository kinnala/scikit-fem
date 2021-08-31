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

        if len(ix) != len(ix0):
            raise ValueError("The length of the index sets used in the "
                             "creation of a periodic mesh should be equal.")

        # reorder vertices: eliminated nodes must have highest index values
        remap = np.empty(mesh.nvertices, dtype=np.int64)
        remap[ix] = np.arange(mesh.nvertices - len(ix),
                              mesh.nvertices,
                              dtype=np.int64)
        oix = np.setdiff1d(np.arange(mesh.nvertices, dtype=np.int64), ix)
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
        tp = reremap[t]

        # check that there are no duplicate indices
        for i in range(tp.shape[0]):
            for j in range(i + 1, tp.shape[0]):
                if (tp[i] == tp[j]).any():
                    raise ValueError("At least one element has a duplicate "
                                     "index.  Usually this means that the "
                                     "element is part of two periodic "
                                     "boundaries which is not allowed.")

        periodic_mesh = cls.from_mesh(reordered_mesh, tp)

        # store reordered mesh and reverse mapping
        periodic_mesh._orig = reordered_mesh
        periodic_mesh._ix = reremap

        return periodic_mesh

    def save(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError

    def element_finder(self, *args, **kwargs):
        raise NotImplementedError
