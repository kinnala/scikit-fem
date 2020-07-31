from typing import Optional, Type, Dict

import numpy as np
from numpy import ndarray

from .mesh2d import Mesh2D, MeshType


class MeshTri(Mesh2D):
    """A mesh consisting of triangular elements.

    The different constructors are:

    - :meth:`~skfem.mesh.MeshTri.__init__`
    - :meth:`~skfem.mesh.MeshTri.load` (requires meshio)
    - :meth:`~skfem.mesh.MeshTri.init_symmetric`
    - :meth:`~skfem.mesh.MeshTri.init_sqsymmetric`
    - :meth:`~skfem.mesh.MeshTri.init_refdom`
    - :meth:`~skfem.mesh.MeshTri.init_tensor`
    - :meth:`~skfem.mesh.MeshTri.init_lshaped`

    Attributes
    ----------
    facets
        An array containing the facet vertices (2 x Nfacets).
    f2t
        An array containing the triangles next to each facet (2 x Nfacets).
        Each column contains two indices to `self.t`.  If the second row is
        zero then the facet is on the boundary.
    t2f
        An array containing the facets belonging to each triangle (3 x Nelems).
        Each column contains three indices to facets.

    """
    refdom: str = "tri"
    brefdom: str = "line"
    meshio_type: str = "triangle"
    name: str = "Triangular"

    t = np.zeros((3, 0), dtype=np.int64)
    t2f = np.zeros((3, 0), dtype=np.int64)

    def __init__(self,
                 p: Optional[ndarray] = None,
                 t: Optional[ndarray] = None,
                 boundaries: Optional[Dict[str, ndarray]] = None,
                 subdomains: Optional[Dict[str, ndarray]] = None,
                 validate: Optional[bool] = True,
                 sort_t: Optional[bool] = True):
        r"""Initialize a triangular mesh.

        If no arguments are given, initializes a mesh with the following
        topology::

            *-------------*
            |\            |
            |  \          |
            |    \        |
            |      \      |
            |        \    |
            |          \  |
            |            \|
            O-------------*

        Parameters
        ----------
        p
            An array containing the points of the mesh (2 x Nvertices).
        t
            An array containing the element connectivity (3 x Nelems), i.e.
            indices to `self.p`.
        subdomains
            Named subsets of elements.
        boundaries
            Named subsets of boundary facets.
        validate
            If `True`, run mesh validity checks.
        sort_t
            If `True`, sort the element connectivity matrix before building
            mappings.

        """
        if p is None and t is None:
            p = np.array([[0., 1., 0., 1.],
                          [0., 0., 1., 1.]], dtype=np.float_)
            t = np.array([[0, 1, 2],
                          [1, 3, 2]], dtype=np.intp).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        self.boundaries = boundaries
        self.subdomains = subdomains
        super(MeshTri, self).__init__()
        if validate:
            self._validate()
        self._build_mappings(sort_t=sort_t)

    @classmethod
    def init_tensor(cls: Type[MeshType],
                    x: ndarray,
                    y: ndarray) -> MeshType:
        r"""Initialize a tensor product mesh.

        The mesh topology is as follows::

            *---------------*
            |'-.|'-.|`'---._|
            |---+---+-------|
            |\  |\  |'.     |
            | \ | \ |  '-.  |
            |  \|  \|     '.|
            *---------------*

        Parameters
        ----------
        x
            The nodal coordinates in dimension x.
        y
            The nodal coordinates in dimension y.

        """
        npx = len(x)
        npy = len(y)
        X, Y = np.meshgrid(np.sort(x), np.sort(y))
        p = np.vstack((X.flatten('F'), Y.flatten('F')))
        ix = np.arange(npx * npy)
        nt = (npx - 1) * (npy - 1)
        t = np.zeros((3, 2 * nt))
        ix = ix.reshape(npy, npx, order='F').copy()
        t[0, :nt] = (ix[0:(npy-1), 0:(npx-1)].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())
        t[1, :nt] = (ix[1:npy, 0:(npx-1)].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())
        t[2, :nt] = (ix[1:npy, 1:npx].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())
        t[0, nt:] = (ix[0:(npy-1), 0:(npx-1)].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())
        t[1, nt:] = (ix[0:(npy-1), 1:npx].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())
        t[2, nt:] = (ix[1:npy, 1:npx].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())

        return cls(p, t.astype(np.int64))

    @classmethod
    def init_symmetric(cls) -> MeshType:
        r"""Initialize a symmetric mesh of the unit square.

        The mesh topology is as follows::

            *------------*
            |\          /|
            |  \      /  |
            |    \  /    |
            |     *      |
            |    /  \    |
            |  /      \  |
            |/          \|
            O------------*

        """
        p = np.array([[0, 1, 1, 0, 0.5],
                      [0, 0, 1, 1, 0.5]], dtype=np.float_)
        t = np.array([[0, 1, 4],
                      [1, 2, 4],
                      [2, 3, 4],
                      [0, 3, 4]], dtype=np.intp).T
        return cls(p, t)

    @classmethod
    def init_sqsymmetric(cls: Type[MeshType]) -> MeshType:
        r"""Initialize a symmetric mesh of the unit square.

        The mesh topology is as follows::

            *------*------*
            |\     |     /|
            |  \   |   /  |
            |    \ | /    |
            *------*------*
            |    / | \    |
            |  /   |   \  |
            |/     |     \|
            O------*------*

        """
        p = np.array([[0, 0.5, 1,   0, 0.5,   1, 0, 0.5, 1],
                      [0, 0,   0, 0.5, 0.5, 0.5, 1,   1, 1]], dtype=np.float_)
        t = np.array([[0, 1, 4],
                      [1, 2, 4],
                      [2, 4, 5],
                      [0, 3, 4],
                      [3, 4, 6],
                      [4, 6, 7],
                      [4, 7, 8],
                      [4, 5, 8]], dtype=np.intp).T
        return cls(p, t)

    @classmethod
    def init_refdom(cls: Type[MeshType]) -> MeshType:
        r"""Initialize a mesh that includes only the reference triangle.

        The mesh topology is as follows::

            *
            |\
            |  \
            |    \
            |      \
            |        \
            |          \
            |            \
            O-------------*

        """
        p = np.array([[0., 1., 0.],
                      [0., 0., 1.]], dtype=np.float_)
        t = np.array([[0, 1, 2]], dtype=np.intp).T
        return cls(p, t)

    @classmethod
    def init_lshaped(cls: Type[MeshType]) -> MeshType:
        r"""Initialize a mesh for the L-shaped domain.

        The mesh topology is as follows::

            *-------*
            | \     |
            |   \   |
            |     \ |
            |-------O-------*
            |     / | \     |
            |   /   |   \   |
            | /     |     \ |
            *---------------*

        """
        p = np.array([[0., 1., 0., -1.,  0., -1., -1.,  1.],
                      [0., 0., 1.,  0., -1., -1.,  1., -1.]], dtype=np.float_)
        t = np.array([[0, 1, 7],
                      [0, 2, 6],
                      [0, 6, 3],
                      [0, 7, 4],
                      [0, 4, 5],
                      [0, 3, 5]], dtype=np.intp).T
        return cls(p, t)

    def _build_mappings(self, sort_t=True):
        # sort to preserve orientations etc.
        if sort_t:
            self.t = np.sort(self.t, axis=0)

        # define facets: in the order (0,1) (1,2) (0,2)
        self.facets = np.sort(np.hstack((
            self.t[[0, 1]],
            self.t[[1, 2]],
            self.t[[0, 2]],
        )), axis=0)

        # get unique facets and build triangle-to-facet
        # mapping: 3 (edges) x Ntris
        tmp = np.ascontiguousarray(self.facets.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        self.facets = self.facets[:, ixa]
        self.t2f = ixb.reshape((3, self.t.shape[1]))

        # build facet-to-triangle mapping: 2 (triangles) x Nedges
        e_tmp = np.hstack((self.t2f[0], self.t2f[1], self.t2f[2]))
        t_tmp = np.tile(np.arange(self.t.shape[1]), (1, 3))[0]

        e_first, ix_first = np.unique(e_tmp, return_index=True)
        # this emulates matlab unique(e_tmp,'last')
        e_last, ix_last = np.unique(e_tmp[::-1], return_index=True)
        ix_last = e_tmp.shape[0] - ix_last - 1

        self.f2t = np.zeros((2, self.facets.shape[1]), dtype=np.int64)
        self.f2t[0, e_first] = t_tmp[ix_first]
        self.f2t[1, e_last] = t_tmp[ix_last]

        # second row to zero if repeated (i.e., on boundary)
        self.f2t[1, np.nonzero(self.f2t[0] == self.f2t[1])[0]] = -1

    def _uniform_refine(self):
        """Perform a single mesh refine."""
        # rename variables
        t = np.copy(self.t)
        p = np.copy(self.p)
        e = self.facets
        sz = p.shape[1]
        t2f = self.t2f + sz

        # new vertices are the midpoints of edges
        new_p = 0.5 * np.vstack((p[0, e[0]] + p[0, e[1]],
                                 p[1, e[0]] + p[1, e[1]]))
        self.p = np.hstack((p, new_p))

        # build new triangle definitions
        self.t = np.hstack((
            np.vstack((t[0], t2f[0], t2f[2])),
            np.vstack((t[1], t2f[0], t2f[1])),
            np.vstack((t[2], t2f[2], t2f[1])),
            np.vstack((t2f[0], t2f[1], t2f[2])),
        ))

        # mapping of indices between old and new facets
        new_facets = np.zeros((2, e.shape[1]), dtype=np.int64)
        ix0 = np.arange(t.shape[1], dtype=np.int64)
        ix1 = ix0 + t.shape[1]
        ix2 = ix0 + 2 * t.shape[1]

        # rebuild mappings
        self._build_mappings()

        # finish mapping of indices between old and new facets
        new_facets[0, t2f[2, :] - sz] = self.t2f[2, ix0]
        new_facets[0, t2f[1, :] - sz] = self.t2f[2, ix1]
        new_facets[0, t2f[0, :] - sz] = self.t2f[0, ix0]
        new_facets[1, t2f[2, :] - sz] = self.t2f[0, ix2]
        new_facets[1, t2f[1, :] - sz] = self.t2f[2, ix2]
        new_facets[1, t2f[0, :] - sz] = self.t2f[0, ix1]

        self._fix_boundaries(new_facets)

    def _adaptive_refine(self, marked):
        """Refine the set of provided elements."""

        def sort_mesh(p, t):
            """Make (0, 2) the longest edge in t."""
            l01 = np.sqrt(np.sum((p[:, t[0]] - p[:, t[1]]) ** 2, axis=0))
            l12 = np.sqrt(np.sum((p[:, t[1]] - p[:, t[2]]) ** 2, axis=0))
            l02 = np.sqrt(np.sum((p[:, t[0]] - p[:, t[2]]) ** 2, axis=0))

            ix01 = (l01 > l02) * (l01 > l12)
            ix12 = (l12 > l01) * (l12 > l02)

            # row swaps
            tmp = t[2, ix01]
            t[2, ix01] = t[1, ix01]
            t[1, ix01] = tmp

            tmp = t[0, ix12]
            t[0, ix12] = t[1, ix12]
            t[1, ix12] = tmp

            return t

        def find_facets(m, marked_elems):
            """Find the facets to split."""
            facets = np.zeros(m.facets.shape[1], dtype=np.int64)
            facets[m.t2f[:, marked_elems].flatten('F')] = 1
            prev_nnz = -1e10

            while np.count_nonzero(facets) - prev_nnz > 0:
                prev_nnz = np.count_nonzero(facets)
                t2facets = facets[m.t2f]
                t2facets[2, t2facets[0, :] + t2facets[1, :] > 0] = 1
                facets[m.t2f[t2facets == 1]] = 1

            return facets

        def split_elements(m, facets):
            """Define new elements."""
            ix = (-1)*np.ones(m.facets.shape[1], dtype=np.int64)
            ix[facets == 1] = (np.arange(np.count_nonzero(facets))
                               + m.p.shape[1])
            ix = ix[m.t2f]

            red =   (ix[0] >= 0) * (ix[1] >= 0) * (ix[2] >= 0)  # noqa
            blue1 = (ix[0] ==-1) * (ix[1] >= 0) * (ix[2] >= 0)  # noqa
            blue2 = (ix[0] >= 0) * (ix[1] ==-1) * (ix[2] >= 0)  # noqa
            green = (ix[0] ==-1) * (ix[1] ==-1) * (ix[2] >= 0)  # noqa
            rest =  (ix[0] ==-1) * (ix[1] ==-1) * (ix[2] ==-1)  # noqa

            # new red elements
            t_red = np.hstack((
                np.vstack((m.t[0, red], ix[0, red], ix[2, red])),
                np.vstack((m.t[1, red], ix[0, red], ix[1, red])),
                np.vstack((m.t[2, red], ix[1, red], ix[2, red])),
                np.vstack(( ix[1, red], ix[2, red], ix[0, red])),  # noqa
            ))

            # new blue elements
            t_blue1 = np.hstack((
                np.vstack((m.t[1, blue1], m.t[0, blue1], ix[2, blue1])),
                np.vstack((m.t[1, blue1],  ix[1, blue1], ix[2, blue1])),  # noqa
                np.vstack((m.t[2, blue1],  ix[2, blue1], ix[1, blue1])),  # noqa
            ))

            t_blue2 = np.hstack((
                np.vstack((m.t[0, blue2], ix[0, blue2],  ix[2, blue2])),  # noqa
                np.vstack(( ix[2, blue2], ix[0, blue2], m.t[1, blue2])),  # noqa
                np.vstack((m.t[2, blue2], ix[2, blue2], m.t[1, blue2])),
            ))

            # new green elements
            t_green = np.hstack((
                np.vstack((m.t[1, green], ix[2, green], m.t[0, green])),
                np.vstack((m.t[2, green], ix[2, green], m.t[1, green])),
            ))

            # new nodes
            p = .5 * (m.p[:, m.facets[0, facets == 1]] +
                      m.p[:, m.facets[1, facets == 1]])

            return np.hstack((m.p, p)),\
                np.hstack((m.t[:, rest], t_red, t_blue1, t_blue2, t_green))

        sorted_mesh = MeshTri(self.p, sort_mesh(self.p, self.t), sort_t=False)
        facets = find_facets(sorted_mesh, marked)
        self.p, self.t = split_elements(sorted_mesh, facets)

        self._build_mappings()

    def mapping(self):
        from skfem.mapping import MappingAffine
        return MappingAffine(self)

    def element_finder(self):
        from matplotlib.tri import Triangulation

        return Triangulation(self.p[0],
                             self.p[1],
                             self.t.T).get_trifinder()
