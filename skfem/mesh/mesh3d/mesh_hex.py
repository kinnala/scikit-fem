from typing import Dict, Optional, Type

import numpy as np
from numpy import ndarray

from .mesh3d import Mesh3D
from ..mesh import MeshType


class MeshHex(Mesh3D):
    """A mesh consisting of hexahedral elements.

    The different constructors are:

    - :meth:`~skfem.mesh.MeshHex.__init__`
    - :meth:`~skfem.mesh.MeshHex.load` (requires meshio)
    - :meth:`~skfem.mesh.MeshHex.init_tensor`

    Attributes
    ----------
    facets
        Each column contains four column indices to `self.p` (4 x Nfacets).
    f2t
        Each column contains a pair of column indices to `self.t`
        or -1 on the second row if the corresponding
        facet is located on the boundary (2 x Nfacets).
    t2f
        Each column contains four indices to `self.facets` (6 x Nelements).
    edges
        Each column corresponds to an edge and contains two indices to
        `self.p` (2 x Nedges).
    t2e
        Each column contains twelve column indices of
        `self.edges` (12 x Nelements).

    """
    refdom: str = "hex"
    brefdom: str = "quad"
    meshio_type: str = "hexahedron"
    name: str = "Hexahedral"

    t = np.zeros((8, 0), dtype=np.int64)
    t2f = np.zeros((6, 0), dtype=np.int64)
    facets = np.zeros((4, 0), dtype=np.int64)
    edges = np.zeros((2, 0), dtype=np.int64)
    t2e = np.zeros((12, 0), dtype=np.int64)

    def __init__(self,
                 p: Optional[ndarray] = None,
                 t: Optional[ndarray] = None,
                 boundaries: Optional[Dict[str, ndarray]] = None,
                 subdomains: Optional[Dict[str, ndarray]] = None,
                 validate=True):
        """Initialise a hexahedral mesh.

        If `t` is provided, order of vertices in each element should match the
        numbering::

            2---6
           /   /|
          4---7 3
          |   |/
          1---5

        where the hidden node is 0.

        Parameters
        ----------
        p
            The points of the mesh (3 x Nvertices).
        t
            The element connectivity (8 x Nelems), i.e. indices to `self.p`.
        subdomains
            Named subsets of elements.
        boundaries
            Named subsets of boundary facets.
        validate
            If `True`, perform mesh validity checks.

        """
        if p is None and t is None:
            p = np.array([[0., 0., 0.],
                          [0., 0., 1.],
                          [0., 1., 0.],
                          [1., 0., 0.],
                          [0., 1., 1.],
                          [1., 0., 1.],
                          [1., 1., 0.],
                          [1., 1., 1.]]).T
            t = np.array([[0, 1, 2, 3, 4, 5, 6, 7]]).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        #
        # TODO fix orientation if p and t is provided. refer to
        # the default mesh for correct orientation
        #
        #   2---6
        #  /   /|
        # 4---7 3
        # |   |/
        # 1---5
        #
        # The hidden node is 0.
        #
        self.p = p
        self.t = t
        self.boundaries = boundaries
        self.subdomains = subdomains
        super(MeshHex, self).__init__()
        if validate:
            self._validate()
        self._build_mappings()

    @classmethod
    def init_refdom(cls: Type[MeshType]):
        """Initialise a mesh of the reference domain."""
        return cls()

    @classmethod
    def init_tensor(cls: Type[MeshType],
                    x: ndarray,
                    y: ndarray,
                    z: ndarray) -> MeshType:
        """Initialise a tensor product mesh.

        Parameters
        ----------
        x : numpy array (1d)
            The nodal coordinates in dimension x
        y : numpy array (1d)
            The nodal coordinates in dimension y
        z : numpy array (1d)
            The nodal coordinates in dimension z

        """
        npx = len(x)
        npy = len(y)
        npz = len(z)
        X, Y, Z = np.meshgrid(np.sort(x), np.sort(y), np.sort(z))
        p = np.vstack((X.flatten('F'), Y.flatten('F'), Z.flatten('F')))
        ix = np.arange(npx * npy * npz)
        ne = (npx - 1) * (npy - 1) * (npz - 1)
        t = np.zeros((8, ne))
        ix = ix.reshape(npy, npx, npz, order='F').copy()
        t[0, :] = (ix[0:(npy - 1), 0:(npx - 1), 0:(npz - 1)]
                   .reshape(ne, 1, order='F')
                   .copy()
                   .flatten())
        t[1, :] = (ix[1:npy, 0:(npx - 1), 0:(npz - 1)]
                   .reshape(ne, 1, order='F')
                   .copy()
                   .flatten())
        t[2, :] = (ix[0:(npy - 1), 1:npx, 0:(npz - 1)]
                   .reshape(ne, 1, order='F')
                   .copy()
                   .flatten())
        t[3, :] = (ix[0:(npy - 1), 0:(npx - 1), 1:npz]
                   .reshape(ne, 1, order='F')
                   .copy()
                   .flatten())
        t[4, :] = (ix[1:npy, 1:npx, 0:(npz - 1)]
                   .reshape(ne, 1, order='F')
                   .copy()
                   .flatten())
        t[5, :] = (ix[1:npy, 0:(npx - 1), 1:npz]
                   .reshape(ne, 1, order='F')
                   .copy()
                   .flatten())
        t[6, :] = (ix[0:(npy - 1), 1:npx, 1:npz]
                   .reshape(ne, 1, order='F')
                   .copy()
                   .flatten())
        t[7, :] = (ix[1:npy, 1:npx, 1:npz]
                   .reshape(ne, 1, order='F')
                   .copy()
                   .flatten())
        return cls(p, t.astype(np.int64))

    def _build_mappings(self):
        """Build element-to-facet, element-to-edges, etc. mappings."""
        self.edges = np.sort(np.hstack((
            self.t[[0, 1]],
            self.t[[0, 2]],
            self.t[[0, 3]],
            self.t[[1, 4]],
            self.t[[1, 5]],
            self.t[[2, 4]],
            self.t[[2, 6]],
            self.t[[3, 5]],
            self.t[[3, 6]],
            self.t[[4, 7]],
            self.t[[5, 7]],
            self.t[[6, 7]],
        )), axis=0)

        # unique edges
        self.edges, ixa, ixb = np.unique(self.edges,
                                         axis=1,
                                         return_index=True,
                                         return_inverse=True)
        self.edges = np.ascontiguousarray(self.edges)

        self.t2e = ixb.reshape((12, self.t.shape[1]))

        # define facets
        self.facets = np.hstack((
            self.t[[0, 1, 4, 2]],
            self.t[[0, 2, 6, 3]],
            self.t[[0, 3, 5, 1]],
            self.t[[2, 4, 7, 6]],
            self.t[[1, 5, 7, 4]],
            self.t[[3, 6, 7, 5]],
        ))

        sorted_facets = np.sort(self.facets, axis=0)

        # unique facets
        sorted_facets, ixa, ixb = np.unique(sorted_facets,
                                            axis=1,
                                            return_index=True,
                                            return_inverse=True)
        self.facets = np.ascontiguousarray(self.facets[:, ixa])

        self.t2f = ixb.reshape((6, self.t.shape[1]))

        # build facet-to-hexa mapping: 2 (hexes) x Nfacets
        e_tmp = np.hstack((self.t2f[0], self.t2f[1],
                           self.t2f[2], self.t2f[3],
                           self.t2f[4], self.t2f[5]))
        t_tmp = np.tile(np.arange(self.t.shape[1]), (1, 6))[0]

        e_first, ix_first = np.unique(e_tmp, return_index=True)
        e_last, ix_last = np.unique(e_tmp[::-1], return_index=True)
        ix_last = e_tmp.shape[0] - ix_last - 1

        self.f2t = np.zeros((2, self.facets.shape[1]), dtype=np.int64)
        self.f2t[0, e_first] = t_tmp[ix_first]
        self.f2t[1, e_last] = t_tmp[ix_last]

        # second row to zero if repeated (i.e., on boundary)
        self.f2t[1, np.nonzero(self.f2t[0, :] == self.f2t[1, :])[0]] = -1

    def _uniform_refine(self):
        """Perform a single mesh refine that halves 'h'. Each hex is
        split into 8."""
        # rename variables
        t = self.t
        p = self.p
        e = self.edges
        f = self.facets
        sz = p.shape[1]
        t2e = self.t2e + sz
        t2f = self.t2f + np.max(t2e) + 1
        # hex middle point
        mid = range(self.t.shape[1]) + np.max(t2f) + 1
        # new vertices are the midpoints of edges ...
        newp1 = 0.5 * np.sum(p[:, e], axis=1)
        # ... midpoints of facets ...
        newp2 = 0.25 * np.sum(p[:, f], axis=1)
        # ... and element middle points
        newp3 = 0.125 * np.sum(p[:, t], axis=1)
        newp = np.hstack((p, newp1, newp2, newp3))
        # build new hex indexing (this requires some serious meditation)
        newt = np.vstack((t[0],
                          t2e[0],
                          t2e[1],
                          t2e[2],
                          t2f[0],
                          t2f[2],
                          t2f[1],
                          mid))
        newt = np.hstack((newt, np.vstack((t2e[0],
                                           t[1],
                                           t2f[0],
                                           t2f[2],
                                           t2e[3],
                                           t2e[4],
                                           mid,
                                           t2f[4]))))
        newt = np.hstack((newt, np.vstack((t2e[1],
                                           t2f[0],
                                           t[2],
                                           t2f[1],
                                           t2e[5],
                                           mid,
                                           t2e[6],
                                           t2f[3]))))
        newt = np.hstack((newt, np.vstack((t2e[2],
                                           t2f[2],
                                           t2f[1],
                                           t[3],
                                           mid,
                                           t2e[7],
                                           t2e[8],
                                           t2f[5]))))
        newt = np.hstack((newt, np.vstack((t2f[0],
                                           t2e[3],
                                           t2e[5],
                                           mid,
                                           t[4],
                                           t2f[4],
                                           t2f[3],
                                           t2e[9]))))
        newt = np.hstack((newt, np.vstack((t2f[2],
                                           t2e[4],
                                           mid,
                                           t2e[7],
                                           t2f[4],
                                           t[5],
                                           t2f[5],
                                           t2e[10],))))
        newt = np.hstack((newt, np.vstack((t2f[1],
                                           mid,
                                           t2e[6],
                                           t2e[8],
                                           t2f[3],
                                           t2f[5],
                                           t[6],
                                           t2e[11]))))
        newt = np.hstack((newt, np.vstack((mid,
                                           t2f[4],
                                           t2f[3],
                                           t2f[5],
                                           t2e[9],
                                           t2e[10],
                                           t2e[11],
                                           t[7]))))
        # update fields
        self.p = newp
        self.t = newt

        self._build_mappings()

    def save(self,
             filename: str,
             point_data: Optional[Dict[str, ndarray]] = None,
             cell_data: Optional[Dict[str, ndarray]] = None):
        """Export the mesh and fields using meshio. (Hexahedron version.)

        Parameters
        ----------
        filename
            The filename for vtk-file.
        point_data
            ndarray for one output or dict for multiple
        cell_data
            ndarray for one output or dict for multiple

        """
        import meshio

        # vtk requires a different ordering
        t = self.t[[0, 3, 6, 2, 1, 5, 7, 4]]

        if point_data is not None:
            if not isinstance(point_data, dict):
                raise ValueError("point_data should be "
                                 "a dictionary of ndarrays.")

        if cell_data is not None:
            if not isinstance(cell_data, dict):
                raise ValueError("cell_data should be "
                                 "a dictionary of ndarrays.")

        cells = {'hexahedron': t.T}
        mesh = meshio.Mesh(self.p.T, cells, point_data, cell_data)
        meshio.write(filename, mesh)

    def mapping(self):
        from skfem.mapping import MappingIsoparametric
        from skfem.element import ElementHex1, ElementQuad1
        return MappingIsoparametric(self, ElementHex1(), ElementQuad1())
