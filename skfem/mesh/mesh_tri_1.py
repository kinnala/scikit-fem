from dataclasses import dataclass, replace
from typing import Type

import numpy as np
from numpy import ndarray
from scipy.spatial import cKDTree

from ..element import Element, ElementTriP1
from .mesh_2d import Mesh2D


@dataclass(repr=False)
class MeshTri1(Mesh2D):
    """A standard first-order triangular mesh."""

    doflocs: ndarray = np.array([[0., 0.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 1.]], dtype=np.float64).T
    t: ndarray = np.array([[0, 1, 2],
                           [1, 3, 2]], dtype=np.int64).T
    elem: Type[Element] = ElementTriP1
    affine: bool = True
    sort_t: bool = True

    @classmethod
    def init_tensor(cls: Type, x: ndarray, y: ndarray):
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
            The nodal coordinates in dimension `x`.
        y
            The nodal coordinates in dimension `y`.

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
    def init_symmetric(cls: Type) -> Mesh2D:
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
        p = np.array([[0., 1., 1., 0., .5],
                      [0., 0., 1., 1., .5]], dtype=np.float64)
        t = np.array([[0, 1, 4],
                      [1, 2, 4],
                      [2, 3, 4],
                      [0, 3, 4]], dtype=np.int64).T
        return cls(p, t)

    @classmethod
    def init_sqsymmetric(cls: Type) -> Mesh2D:
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
        p = np.array([[0., .5, 1., 0., .5, 1., 0., .5, 1.],
                      [0., 0., 0., .5, .5, .5, 1., 1., 1.]], dtype=np.float64)
        t = np.array([[0, 1, 4],
                      [1, 2, 4],
                      [2, 4, 5],
                      [0, 3, 4],
                      [3, 4, 6],
                      [4, 6, 7],
                      [4, 7, 8],
                      [4, 5, 8]], dtype=np.int64).T
        return cls(p, t)

    @classmethod
    def init_lshaped(cls: Type) -> Mesh2D:
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
                      [0., 0., 1.,  0., -1., -1.,  1., -1.]], dtype=np.float64)
        t = np.array([[0, 1, 7],
                      [0, 2, 6],
                      [0, 6, 3],
                      [0, 7, 4],
                      [0, 4, 5],
                      [0, 3, 5]], dtype=np.int64).T
        return cls(p, t)

    @classmethod
    def init_circle(cls: Type,
                    nrefs: int = 3) -> Mesh2D:
        r"""Initialize a circle mesh.

        Works by repeatedly refining the following mesh and moving
        new nodes to the boundary::

                   *
                 / | \
               /   |   \
             /     |     \
            *------O------*
             \     |     /
               \   |   /
                 \ | /
                   *

        Parameters
        ----------
        nrefs
            Number of refinements, by default 3.

        """
        p = np.array([[0., 0.],
                      [1., 0.],
                      [0., 1.],
                      [-1., 0.],
                      [0., -1.]], dtype=np.float64).T
        t = np.array([[0, 1, 2],
                      [0, 1, 4],
                      [0, 2, 3],
                      [0, 3, 4]], dtype=np.int64).T
        m = cls(p, t)
        for _ in range(nrefs):
            m = m.refined()
            D = m.boundary_nodes()
            tmp = m.p
            tmp[:, D] = tmp[:, D] / np.linalg.norm(tmp[:, D], axis=0)
            m = replace(m, doflocs=tmp)
        return m

    def _uniform(self):

        p = self.doflocs
        t = self.t
        sz = p.shape[1]
        t2f = self.t2f.copy() + sz
        return replace(
            self,
            doflocs=np.hstack((p, p[:, self.facets].mean(axis=1))),
            t=np.hstack((
                np.vstack((t[0], t2f[0], t2f[2])),
                np.vstack((t[1], t2f[0], t2f[1])),
                np.vstack((t[2], t2f[2], t2f[1])),
                np.vstack((t2f[0], t2f[1], t2f[2])),
            )),
            _boundaries=None,
            _subdomains=None,
        )

    @staticmethod
    def _adaptive_sort_mesh(p, t):
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

    @staticmethod
    def _adaptive_find_facets(m, marked_elems):
        """Find the facets to split."""
        facets = np.zeros(m.facets.shape[1], dtype=np.int64)
        facets[m.t2f[:, marked_elems].flatten('F')] = 1
        prev_nnz = -1e10

        while np.count_nonzero(facets) - prev_nnz > 0:
            prev_nnz = np.count_nonzero(facets)
            t2facets = facets[m.t2f]
            t2facets[2, t2facets[0] + t2facets[1] > 0] = 1
            facets[m.t2f[t2facets == 1]] = 1

        return facets

    @staticmethod
    def _adaptive_split_elements(m, facets):
        """Define new elements."""
        ix = (-1) * np.ones(m.facets.shape[1], dtype=np.int64)
        ix[facets == 1] = (np.arange(np.count_nonzero(facets))
                           + m.p.shape[1])
        ix = ix[m.t2f]

        red = (ix[0] >= 0) * (ix[1] >= 0) * (ix[2] >= 0)
        blue1 = (ix[0] == -1) * (ix[1] >= 0) * (ix[2] >= 0)
        blue2 = (ix[0] >= 0) * (ix[1] == -1) * (ix[2] >= 0)
        green = (ix[0] == -1) * (ix[1] == -1) * (ix[2] >= 0)
        rest = (ix[0] == -1) * (ix[1] == -1) * (ix[2] == -1)

        # new red elements
        t_red = np.hstack((
            np.vstack((m.t[0, red], ix[0, red], ix[2, red])),
            np.vstack((m.t[1, red], ix[0, red], ix[1, red])),
            np.vstack((m.t[2, red], ix[1, red], ix[2, red])),
            np.vstack((ix[1, red], ix[2, red], ix[0, red])),
        ))

        # new blue elements
        t_blue1 = np.hstack((
            np.vstack((m.t[1, blue1], m.t[0, blue1], ix[2, blue1])),
            np.vstack((m.t[1, blue1], ix[1, blue1], ix[2, blue1])),
            np.vstack((m.t[2, blue1], ix[2, blue1], ix[1, blue1])),
        ))

        t_blue2 = np.hstack((
            np.vstack((m.t[0, blue2], ix[0, blue2], ix[2, blue2])),
            np.vstack((ix[2, blue2], ix[0, blue2], m.t[1, blue2])),
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

        return (
            np.hstack((m.p, p)),
            np.hstack((m.t[:, rest], t_red, t_blue1, t_blue2, t_green)),
        )

    def _adaptive(self, marked):

        sorted_mesh = replace(
            self,
            t=self._adaptive_sort_mesh(self.p, self.t),
            sort_t=False,
        )
        facets = self._adaptive_find_facets(sorted_mesh, marked)
        doflocs, t = self._adaptive_split_elements(sorted_mesh, facets)

        return replace(
            self,
            doflocs=doflocs,
            t=t,
            _boundaries=None,
            _subdomains=None,
        )

    def orientation(self):

        mapping = self._mapping()
        return ((mapping.detDF(np.array([[0], [0]])) < 0)
                .flatten()
                .astype(np.int64))

    def oriented(self):

        flip = np.nonzero(self.orientation())[0]
        t = self.t.copy()
        t0 = t[0, flip]
        t1 = t[1, flip]
        t[0, flip] = t1
        t[1, flip] = t0

        return replace(
            self,
            t=t,
            sort_t=False,
        )

    def __mul__(self, other):

        from .mesh_wedge_1 import MeshWedge1
        from .mesh_line_1 import MeshLine1

        if isinstance(other, MeshLine1):
            points = np.zeros((3, 0), dtype=np.float64)
            wedges = np.zeros((6, 0), dtype=np.int64)
            diff = 0
            for i, p in enumerate(np.sort(other.p[0])):
                points = np.hstack((
                    points,
                    np.vstack((self.p,
                               np.array(self.p.shape[1] * [p])))
                ))
                if i == len(other.p[0]) - 1:
                    pass
                else:
                    wedges = np.hstack((
                        wedges,
                        np.vstack((self.t + diff,
                                   self.t + self.nvertices + diff))
                    ))
                diff += self.nvertices
            return MeshWedge1(points, wedges)

        raise NotImplementedError

    def element_finder(self, mapping=None):

        if mapping is None:
            mapping = self._mapping()

        if not hasattr(self, '_cached_tree'):
            self._cached_tree = cKDTree(np.mean(self.p[:, self.t], axis=1).T)

        tree = self._cached_tree
        nelems = self.t.shape[1]

        def finder(x, y, _search_all=False):

            if not _search_all:
                ix = tree.query(np.array([x, y]).T,
                                min(5, nelems))[1].flatten()
                _, ix_ind = np.unique(ix, return_index=True)
                ix = ix[np.sort(ix_ind)]
            else:
                ix = np.arange(nelems, dtype=np.int64)

            X = mapping.invF(np.array([x, y])[:, None], ix)
            inside = ((X[0] >= 0) *
                      (X[1] >= 0) *
                      (1 - X[0] - X[1] >= 0))

            if not inside.max(axis=0).all():
                if _search_all:
                    raise ValueError("Point is outside of the mesh.")
                return finder(x, y, _search_all=True)

            return np.array([ix[inside.argmax(axis=0)]]).flatten()

        return finder
