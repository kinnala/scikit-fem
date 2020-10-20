from typing import Type, Optional, Dict

import numpy as np
from numpy import ndarray
from scipy.spatial import cKDTree

from .mesh3d import Mesh3D


class MeshTet(Mesh3D):
    """A mesh consisting of tetrahedral elements.

    The different constructors are:

    - :meth:`~skfem.mesh.MeshTet.__init__`
    - :meth:`~skfem.mesh.MeshTet.load` (requires meshio)
    - :meth:`~skfem.mesh.MeshTet.init_tensor`

    Attributes
    ----------
    facets
        Each column contains a triplet of column indices to `self.p`
        (3 x Nfacets).  Order: (0, 1, 2) (0, 1, 3) (0, 2, 3) (1, 2, 3).
    f2t
        Each column contains a pair of column indices to `self.t` or
        -1 on the second row if the facet is located on the boundary
        (2 x Nfacets).
    t2f
        Each column contains four indices to `self.facets` (4 x Nelements).
    edges
        Each column corresponds to an edge and contains two indices to
        `self.p` (2 x Nedges).
        Order: (0, 1) (1, 2) (0, 2) (0, 3) (1, 3) (2, 3).
    t2e
        Each column contains six indices to `self.edges` (6 x Nelements).

    """
    refdom: str = "tet"
    brefdom: str = "tri"
    meshio_type: str = "tetra"
    name: str = "Tetrahedral"

    t = np.zeros((4, 0), dtype=np.int64)
    t2f = np.zeros((4, 0), dtype=np.int64)
    facets = np.zeros((3, 0), dtype=np.int64)
    edges = np.zeros((2, 0), dtype=np.int64)
    t2e = np.zeros((6, 0), dtype=np.int64)

    def __init__(self,
                 p: Optional[ndarray] = None,
                 t: Optional[ndarray] = None,
                 boundaries: Optional[Dict[str, ndarray]] = None,
                 subdomains: Optional[Dict[str, ndarray]] = None,
                 validate=True):
        """Initialize a tetrahedral mesh.

        Parameters
        ----------
        p
            The points of the mesh (3 x Nvertices).
        t
            The element connectivity (6 x Nelems), i.e. indices to `self.p`.
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
            t = np.array([[0, 1, 2, 3],
                          [3, 5, 1, 7],
                          [2, 3, 6, 7],
                          [2, 3, 1, 7],
                          [1, 2, 4, 7]]).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        self.boundaries = boundaries
        self.subdomains = subdomains
        super(MeshTet, self).__init__()
        if validate:
            self._validate()
        self.enable_facets = True
        self._build_mappings()

    @classmethod
    def init_refdom(cls: Type) -> Mesh3D:
        """Initialise a mesh of the reference domain."""
        p = np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]]).T
        t = np.array([[0, 1, 2, 3]]).T
        return cls(p, t)

    @classmethod
    def init_tensor(cls: Type,
                    x: ndarray,
                    y: ndarray,
                    z: ndarray) -> Mesh3D:
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
        t[0] = (ix[0:(npy - 1), 0:(npx - 1), 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[1] = (ix[1:npy, 0:(npx - 1), 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[2] = (ix[0:(npy - 1), 1:npx, 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[3] = (ix[0:(npy - 1), 0:(npx - 1), 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[4] = (ix[1:npy, 1:npx, 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[5] = (ix[1:npy, 0:(npx - 1), 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[6] = (ix[0:(npy - 1), 1:npx, 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[7] = (ix[1:npy, 1:npx, 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())

        T = np.zeros((4, 0))
        T = np.hstack((T, t[[0, 1, 5, 7]]))
        T = np.hstack((T, t[[0, 1, 4, 7]]))
        T = np.hstack((T, t[[0, 2, 4, 7]]))
        T = np.hstack((T, t[[0, 3, 5, 7]]))
        T = np.hstack((T, t[[0, 2, 6, 7]]))
        T = np.hstack((T, t[[0, 3, 6, 7]]))

        return cls(p, T.astype(np.int64))

    @classmethod
    def init_ball(cls: Type,
                  Nrefs: int = 3) -> Mesh3D:
        r"""Initialize a ball mesh.

        Parameters
        ----------
        Nrefs
            Number of refinements, by default 3.

        """
        p = np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.],
                      [-1., 0., 0.],
                      [0., -1., 0.],
                      [0., 0., -1.]]).T
        t = np.array([[0, 1, 2, 3],
                      [0, 4, 5, 6],
                      [0, 1, 2, 6],
                      [0, 1, 3, 5],
                      [0, 2, 3, 4],
                      [0, 4, 5, 3],
                      [0, 4, 6, 2],
                      [0, 5, 6, 1]], dtype=np.intp).T
        m = cls(p, t)
        for _ in range(Nrefs):
            m.refine()
            D = m.boundary_nodes()
            m.p[:, D] = m.p[:, D] / np.linalg.norm(m.p[:, D], axis=0)
        return m

    def _build_mappings(self):
        """Build element-to-facet, element-to-edges, etc. mappings."""
        # define edges: in the order (0,1) (1,2) (0,2) (0,3) (1,3) (2,3)
        self.edges = np.sort(np.hstack((
            self.t[[0, 1]],
            self.t[[1, 2]],
            self.t[[0, 2]],
            self.t[[0, 3]],
            self.t[[1, 3]],
            self.t[[2, 3]]
        )), axis=0)

        # unique edges
        self.edges, ixa, ixb = np.unique(self.edges,
                                         axis=1,
                                         return_index=True,
                                         return_inverse=True)
        self.edges = np.ascontiguousarray(self.edges)

        self.t2e = ixb.reshape((6, self.t.shape[1]))

        # define facets
        if self.enable_facets:
            self.facets = np.sort(np.hstack((
                self.t[[0, 1, 2]],
                self.t[[0, 1, 3]],
                self.t[[0, 2, 3]],
                self.t[[1, 2, 3]]
            )), axis=0)

            # unique facets
            self.facets, ixa, ixb = np.unique(self.facets,
                                              axis=1,
                                              return_index=True,
                                              return_inverse=True)
            self.facets = np.ascontiguousarray(self.facets)

            self.t2f = ixb.reshape((4, self.t.shape[1]))

            # build facet-to-tetra mapping: 2 (tets) x Nfacets
            e_tmp = np.hstack((self.t2f[0], self.t2f[1],
                               self.t2f[2], self.t2f[3]))
            t_tmp = np.tile(np.arange(self.t.shape[1]), (1, 4))[0]

            e_first, ix_first = np.unique(e_tmp, return_index=True)
            # this emulates matlab unique(e_tmp,'last')
            e_last, ix_last = np.unique(e_tmp[::-1], return_index=True)
            ix_last = e_tmp.shape[0] - ix_last-1

            self.f2t = np.zeros((2, self.facets.shape[1]), dtype=np.int64)
            self.f2t[0, e_first] = t_tmp[ix_first]
            self.f2t[1, e_last] = t_tmp[ix_last]

            # second row to zero if repeated (i.e., on boundary)
            self.f2t[1, np.nonzero(self.f2t[0, :] == self.f2t[1, :])[0]] = -1

    def refine(self, N=None):
        """Refine the mesh, tetrahedral optimization.

        Parameters
        ----------
        N : (optional) int
            Perform N refinements.

        """
        if N is None:
            return self._uniform_refine()
        else:
            self.enable_facets = False
            for itr in range(N-1):
                self._uniform_refine()
            self.enable_facets = True
            self._uniform_refine()

    def _uniform_refine(self):
        """Perform a single mesh refine.

        Let the nodes of a tetrahedron be numbered as 0, 1, 2 and 3.
        It is assumed that the edges in self.t2e are given in the order

          I=(0,1), II=(1,2), III=(0,2), IV=(0,3), V=(1,3), VI=(2,3)

        by self._build_mappings(). Let I denote the midpoint of the edge
        (0,1), II denote the midpoint of the edge (1,2), etc. Then each
        tetrahedron is split into eight smaller subtetrahedra as follows.

        The first four subtetrahedra have the following nodes:

          1. (0,I,III,IV)
          2. (1,I,II,V)
          3. (2,II,III,VI)
          4. (3,IV,V,VI)

        The remaining middle-portion of the original tetrahedron consists
        of a union of two mirrored pyramids. This bi-pyramid can be splitted
        into four tetrahedra in a three different ways by connecting the
        midpoints of two opposing edges (there are three different pairs
        of opposite edges).

        For each tetrahedra in the original mesh, we split the bi-pyramid
        in such a way that the connection between the opposite edges
        is shortest. This minimizes the shape-regularity constant of
        the resulting mesh family.

        """
        # rename variables
        t = self.t
        p = self.p
        e = self.edges
        sz = p.shape[1]
        t2e = self.t2e + sz
        # new vertices are the midpoints of edges
        newp = .5 * np.vstack((p[0, e[0]] + p[0, e[1]],
                               p[1, e[0]] + p[1, e[1]],
                               p[2, e[0]] + p[2, e[1]]))
        newp = np.hstack((p, newp))
        # new tets
        newt = np.vstack((t[0], t2e[0], t2e[2], t2e[3]))
        newt = np.hstack((newt, np.vstack((t[1], t2e[0], t2e[1], t2e[4]))))
        newt = np.hstack((newt, np.vstack((t[2], t2e[1], t2e[2], t2e[5]))))
        newt = np.hstack((newt, np.vstack((t[3], t2e[3], t2e[4], t2e[5]))))
        # compute middle pyramid diagonal lengths and choose shortest
        d1 = ((newp[0, t2e[2]] - newp[0, t2e[4]]) ** 2 +
              (newp[1, t2e[2]] - newp[1, t2e[4]]) ** 2)
        d2 = ((newp[0, t2e[1]] - newp[0, t2e[3]]) ** 2 +
              (newp[1, t2e[1]] - newp[1, t2e[3]]) ** 2)
        d3 = ((newp[0, t2e[0]] - newp[0, t2e[5]]) ** 2 +
              (newp[1, t2e[0]] - newp[1, t2e[5]]) ** 2)
        I1 = d1 < d2
        I2 = d1 < d3
        I3 = d2 < d3
        c1 = I1 * I2
        c2 = (~I1) * I3
        c3 = (~I2) * (~I3)
        # splitting the pyramid in the middle.
        # diagonals are [2,4], [1,3] and [0,5]
        # CASE 1: diagonal [2,4]
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[0, c1], t2e[1, c1]))))
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[0, c1], t2e[3, c1]))))
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[1, c1], t2e[5, c1]))))
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[3, c1], t2e[5, c1]))))
        # CASE 2: diagonal [1,3]
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[0, c2], t2e[4, c2]))))
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[4, c2], t2e[5, c2]))))
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[5, c2], t2e[2, c2]))))
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[2, c2], t2e[0, c2]))))
        # CASE 3: diagonal [0,5]
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[1, c3], t2e[4, c3]))))
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[4, c3], t2e[3, c3]))))
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[3, c3], t2e[2, c3]))))
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[2, c3], t2e[1, c3]))))
        # update fields
        self.p = newp
        self.t = newt

        self._build_mappings()

    def element_finder(self, mapping=None):
        """Return a function handle from location to element index.

        Parameters
        ----------
        mapping
            The affine mapping for the mesh.

        """
        if mapping is None:
            raise NotImplementedError("Mapping must be provided.")

        tree = cKDTree(np.mean(self.p[:, self.t], axis=1).T)

        def finder(x, y, z):
            ix = tree.query(np.array([x, y, z]).T, 5)[1].flatten()
            X = mapping.invF(np.array([x, y, z])[:, None], ix)
            inside = (
                (X[0] >= 0)
                * (X[1] >= 0)
                * (X[2] >= 0)
                * (1 - X[0] - X[1] - X[2] >= 0)
            )
            return np.array([ix[np.argmax(inside, axis=0)]]).flatten()

        return finder

    def shapereg(self):
        """Return the largest shape-regularity constant."""
        def edgelen(n):
            return np.sqrt(np.sum((self.p[:, self.edges[0, self.t2e[n]]] -
                                   self.p[:, self.edges[1, self.t2e[n]]]) ** 2,
                                  axis=0))
        edgelenmat = np.vstack(tuple(edgelen(i) for i in range(6)))
        return np.max(np.max(edgelenmat, axis=0) / np.min(edgelenmat, axis=0))

    def mapping(self):
        from skfem.mapping import MappingAffine
        return MappingAffine(self)
