from dataclasses import dataclass, replace
from typing import Type

import numpy as np
from numpy import ndarray
from scipy.spatial import cKDTree

from ..element import Element, ElementTetP1
from .mesh_3d import Mesh3D


@dataclass(repr=False)
class MeshTet1(Mesh3D):
    """A standard first-order tetrahedral mesh."""

    doflocs: ndarray = np.array([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 1., 1.],
                                 [1., 0., 1.],
                                 [1., 1., 0.],
                                 [1., 1., 1.]], dtype=np.float64).T
    t: ndarray = np.array([[0, 1, 2, 3],
                           [3, 5, 1, 7],
                           [2, 3, 6, 7],
                           [2, 3, 1, 7],
                           [1, 2, 4, 7]], dtype=np.int64).T
    elem: Type[Element] = ElementTetP1
    affine: bool = True

    def element_finder(self, mapping=None):

        if mapping is None:
            mapping = self._mapping()

        if not hasattr(self, '_cached_tree'):
            self._cached_tree = cKDTree(np.mean(self.p[:, self.t], axis=1).T)

        tree = self._cached_tree
        nelems = self.t.shape[1]

        def finder(x, y, z, _search_all=False):

            if not _search_all:
                ix = tree.query(np.array([x, y, z]).T,
                                min(10, nelems))[1].flatten()
                _, ix_ind = np.unique(ix, return_index=True)
                ix = ix[np.sort(ix_ind)]
            else:
                ix = np.arange(nelems, dtype=np.int64)

            X = mapping.invF(np.array([x, y, z])[:, None], ix)
            inside = ((X[0] >= 0) *
                      (X[1] >= 0) *
                      (X[2] >= 0) *
                      (1 - X[0] - X[1] - X[2] >= 0))

            if not inside.max(axis=0).all():
                if _search_all:
                    raise ValueError("Point is outside of the mesh.")
                return finder(x, y, z, _search_all=True)

            return np.array([ix[inside.argmax(axis=0)]]).flatten()

        return finder

    def _uniform(self):
        t = self.t
        p = self.p
        sz = p.shape[1]
        t2e = self.t2e.copy() + sz

        # new vertices are the midpoints of edges
        newp = np.hstack((p, p[:, self.edges].mean(axis=1)))

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

        # splitting the pyramid in the middle;
        # diagonals are [2,4], [1,3] and [0,5]
        newt = np.hstack((
            np.vstack((t[0], t2e[0], t2e[2], t2e[3])),
            np.vstack((t[1], t2e[0], t2e[1], t2e[4])),
            np.vstack((t[2], t2e[1], t2e[2], t2e[5])),
            np.vstack((t[3], t2e[3], t2e[4], t2e[5])),
            np.vstack((t2e[2, c1], t2e[4, c1], t2e[0, c1], t2e[1, c1])),
            np.vstack((t2e[2, c1], t2e[4, c1], t2e[0, c1], t2e[3, c1])),
            np.vstack((t2e[2, c1], t2e[4, c1], t2e[1, c1], t2e[5, c1])),
            np.vstack((t2e[2, c1], t2e[4, c1], t2e[3, c1], t2e[5, c1])),
            np.vstack((t2e[1, c2], t2e[3, c2], t2e[0, c2], t2e[4, c2])),
            np.vstack((t2e[1, c2], t2e[3, c2], t2e[4, c2], t2e[5, c2])),
            np.vstack((t2e[1, c2], t2e[3, c2], t2e[5, c2], t2e[2, c2])),
            np.vstack((t2e[1, c2], t2e[3, c2], t2e[2, c2], t2e[0, c2])),
            np.vstack((t2e[0, c3], t2e[5, c3], t2e[1, c3], t2e[4, c3])),
            np.vstack((t2e[0, c3], t2e[5, c3], t2e[4, c3], t2e[3, c3])),
            np.vstack((t2e[0, c3], t2e[5, c3], t2e[3, c3], t2e[2, c3])),
            np.vstack((t2e[0, c3], t2e[5, c3], t2e[2, c3], t2e[1, c3])),
        ))

        return replace(
            self,
            doflocs=newp,
            t=newt,
            _boundaries=None,
            _subdomains=None,
        )

    @staticmethod
    def _adaptive_sort_mesh(p, t, marked):
        """Make (0, 1) the longest edge in t for marked."""

        # add noise so that there are no edges with the same length
        np.random.seed(1337)
        p = p.copy() + 1e-5 * np.random.random(p.shape)

        l01 = np.sqrt(np.sum((p[:, t[0, marked]] - p[:, t[1, marked]]) ** 2,
                             axis=0))
        l12 = np.sqrt(np.sum((p[:, t[1, marked]] - p[:, t[2, marked]]) ** 2,
                             axis=0))
        l02 = np.sqrt(np.sum((p[:, t[0, marked]] - p[:, t[2, marked]]) ** 2,
                             axis=0))
        l03 = np.sqrt(np.sum((p[:, t[0, marked]] - p[:, t[3, marked]]) ** 2,
                             axis=0))
        l13 = np.sqrt(np.sum((p[:, t[1, marked]] - p[:, t[3, marked]]) ** 2,
                             axis=0))
        l23 = np.sqrt(np.sum((p[:, t[2, marked]] - p[:, t[3, marked]]) ** 2,
                             axis=0))

        # indices where (1, 2) is the longest etc.
        ix12 = ((l12 > l01)
                * (l12 > l02)
                * (l12 > l03)
                * (l12 > l13)
                * (l12 > l23))
        ix02 = ((l02 > l01)
                * (l02 > l12)
                * (l02 > l03)
                * (l02 > l13)
                * (l02 > l23))
        ix03 = ((l03 > l01)
                * (l03 > l12)
                * (l03 > l02)
                * (l03 > l13)
                * (l03 > l23))
        ix13 = ((l13 > l01)
                * (l13 > l12)
                * (l13 > l02)
                * (l13 > l03)
                * (l13 > l23))
        ix23 = ((l23 > l01)
                * (l23 > l12)
                * (l23 > l02)
                * (l23 > l03)
                * (l23 > l13))

        # flip edges
        T = t.copy()
        T[:, marked[ix02]] = t[:, marked[ix02]][[2, 0, 1, 3]]
        T[:, marked[ix03]] = t[:, marked[ix03]][[0, 3, 1, 2]]
        T[:, marked[ix12]] = t[:, marked[ix12]][[1, 2, 0, 3]]
        T[:, marked[ix13]] = t[:, marked[ix13]][[1, 3, 2, 0]]
        T[:, marked[ix23]] = t[:, marked[ix23]][[3, 2, 1, 0]]

        return T

    def _find_nz(self, rows, cols, shape, transform=None):
        """Find nonzero entries from the incidence matrix after transform."""
        from scipy.sparse import coo_matrix, find
        rows = rows.flatten('C')
        cols = cols.flatten('C')
        inc = coo_matrix((np.ones(len(rows)), (rows, cols)),
                         shape=shape).tocsr()
        if transform is not None:
            inc = transform(inc)
        inc.eliminate_zeros()
        return find(inc)[:2]

    def _adaptive(self, marked):
        """Longest edge bisection."""
        if isinstance(marked, list):
            marked = np.array(marked, dtype=np.int64)
        nt = self.t.shape[1]
        nv = self.p.shape[1]
        p = np.zeros((3, 9 * nv), dtype=np.float64)
        t = np.zeros((4, 4 * nt), dtype=np.int64)
        p[:, :self.p.shape[1]] = self.p.copy()
        t[:, :self.t.shape[1]] = self.t.copy()

        gen = np.zeros(nv + 6 * nt, dtype=np.int8)
        nonconf = np.ones(8 * nv, dtype=np.int8)
        split_edge = np.zeros((3, 8 * nv), dtype=np.int64)
        ns = 0

        while len(marked) > 0:
            nm = len(marked)
            tnew = np.zeros(nm, dtype=np.int64)
            ix = np.arange(nm, dtype=np.int64)
            t = self._adaptive_sort_mesh(p, t, marked)
            t0, t1, t2, t3 = t[:, marked]

            if ns > 0:
                nonconf_edge = np.nonzero(nonconf[:ns])[0]
                i, j = self._find_nz(
                    split_edge[:2, nonconf_edge],
                    np.vstack((split_edge[2, nonconf_edge],) * 2),
                    (nv, nv),
                    lambda I: I[t0].multiply(I[t1])
                )
                tnew[i] = j
                ix = np.nonzero(tnew == 0)[0]

            if len(ix) > 0:
                i, j = self._find_nz(
                    *np.sort(np.vstack((t0[ix], t1[ix])), axis=0),
                    (nv, nv),
                )
                nn = len(i)
                nix = slice(ns, ns + nn)
                split_edge[0, nix] = i
                split_edge[1, nix] = j
                split_edge[2, nix] = np.arange(nv, nv + nn, dtype=np.int64)

                # add new points
                p[:, nv:(nv + nn)] = .5 * (p[:, i] + p[:, j])
                nv += nn
                i, j = self._find_nz(
                    split_edge[:2, nix],
                    np.vstack((split_edge[2, nix],) * 2),
                    (nv, nv),
                    lambda I: I[t0].multiply(I[t1])
                )
                tnew[i] = j
                ns += nn

            ix = np.nonzero(gen[tnew] == 0)[0]
            gen[tnew[ix]] = np.max(gen[t[:, marked[ix]]], axis=0) + 1

            # add new elements
            t[:, marked] = np.vstack((t3, t0, t2, tnew))
            t[:, nt:(nt + nm)] = np.vstack((t2, t1, t3, tnew))
            nt += nm

            check = np.nonzero(nonconf[:ns])[0]
            nonconf[check] = 0
            check_node = np.zeros(nv, dtype=np.int64)
            check_node[split_edge[:2, check]] = 1
            check_elem = np.nonzero(check_node[t[:, :nt]].sum(axis=0))[0]

            i, j = self._find_nz(
                t[:, check_elem],
                np.vstack((check_elem,) * 4),
                (nv, nt),
                lambda I: (I[split_edge[0, check]]
                           .multiply(I[split_edge[1, check]]))
            )
            nonconf[check[i]] = 1
            marked = np.unique(j)

        return replace(
            self,
            doflocs=p[:, :nv],
            t=t[:, :nt],
        )

    @classmethod
    def init_tensor(cls: Type,
                    x: ndarray,
                    y: ndarray,
                    z: ndarray):
        """Initialize a tensor product mesh.

        Parameters
        ----------
        x
            The nodal coordinates in dimension `x`.
        y
            The nodal coordinates in dimension `y`.
        z
            The nodal coordinates in dimension `z`.

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
                  nrefs: int = 3):
        """Initialize a ball mesh.

        Parameters
        ----------
        nrefs
            Number of refinements, by default 3.

        """
        p = np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.],
                      [-1., 0., 0.],
                      [0., -1., 0.],
                      [0., 0., -1.]], dtype=np.float64).T
        t = np.array([[0, 1, 2, 3],
                      [0, 4, 5, 6],
                      [0, 1, 2, 6],
                      [0, 1, 3, 5],
                      [0, 2, 3, 4],
                      [0, 4, 5, 3],
                      [0, 4, 6, 2],
                      [0, 5, 6, 1]], dtype=np.int64).T
        m = cls(p, t)
        for _ in range(nrefs):
            m = m.refined()
            D = m.boundary_nodes()
            tmp = m.p
            tmp[:, D] = tmp[:, D] / np.linalg.norm(tmp[:, D], axis=0)
            m = replace(m, doflocs=tmp)
        return m
