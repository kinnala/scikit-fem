from dataclasses import replace

import numpy as np


class MeshDG:

    @classmethod
    def init_tensor(cls,
                    *args,
                    periodic=[]):
        mesh = cls.__bases__[-1].init_tensor(*args)
        ix = np.empty((0,), dtype=np.int32)
        ix0 = np.empty((0,), dtype=np.int32)
        for dim in periodic:
            argmin = args[dim].min()
            argmax = args[dim].max()
            ix = np.concatenate((
                ix,
                mesh.nodes_satisfying(lambda x: argmin == x[dim]),
            ))
            ix0 = np.concatenate((
                ix0,
                mesh.nodes_satisfying(lambda x: argmax == x[dim]),
            ))
        ix, uniq = np.unique(ix, return_index=True)
        ix0 = ix0[uniq]
        for k, idx in enumerate(ix):
            ix0[ix0 == idx] = ix0[k]
        return cls.periodic(mesh, ix, ix0)

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
        remap = np.empty(mesh.nvertices, dtype=np.int32)
        remap[ix] = np.arange(mesh.nvertices - len(ix),
                              mesh.nvertices,
                              dtype=np.int32)
        oix = np.setdiff1d(np.arange(mesh.nvertices, dtype=np.int32), ix)
        remap[oix] = np.arange(mesh.nvertices - len(ix), dtype=np.int32)

        doflocs = np.hstack((mesh.doflocs[:, oix], mesh.doflocs[:, ix]))
        t = remap[mesh.t]

        reordered_mesh = replace(
            mesh,
            doflocs=doflocs,
            t=t,
            sort_t=False,
        )

        # make periodic
        reremap = np.arange(mesh.nvertices, dtype=np.int32)
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

    def draw(self, *args, **kwargs):
        from ..assembly import CellBasis
        return CellBasis(self, self.elem()).draw(*args, **kwargs)
