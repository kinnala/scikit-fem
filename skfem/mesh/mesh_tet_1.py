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

        def finder(x, y, z):

            ix = tree.query(np.array([x, y, z]).T,
                            min(10, nelems))[1].flatten()
            _, ix_ind = np.unique(ix, return_index=True)
            ix = ix[np.sort(ix_ind)]

            X = mapping.invF(np.array([x, y, z])[:, None], ix)
            inside = ((X[0] >= 0) *
                      (X[1] >= 0) *
                      (X[2] >= 0) *
                      (1 - X[0] - X[1] - X[2] >= 0))
            elems = np.argmax(inside, axis=0)

            return np.array([ix[elems]]).flatten()

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
