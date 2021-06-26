from dataclasses import dataclass, replace
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementHex1
from .mesh_3d import Mesh3D
from .mesh_tet_1 import MeshTet1


@dataclass(repr=False)
class MeshHex1(Mesh3D):
    """A standard first-order hexahedral mesh.

    If ``t`` is provided, order of vertices in each element should match the
    numbering::

            2---6
           /   /|
          4---7 3
          |   |/
          1---5

    """

    doflocs: ndarray = np.array([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 1., 1.],
                                 [1., 0., 1.],
                                 [1., 1., 0.],
                                 [1., 1., 1.]], dtype=np.float64).T
    t: ndarray = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64).T
    elem: Type[Element] = ElementHex1

    def _init_facets(self):
        """Initialize ``self.facets`` without sorting"""
        self._facets, self._t2f = self.build_entities(
            self.t,
            self.elem.refdom.facets,
            sort=False,
        )

    def _uniform(self):
        p = self.doflocs
        t = self.t
        sz = p.shape[1]
        t2e = self.t2e.copy() + sz
        t2f = self.t2f.copy() + np.max(t2e) + 1
        mid = np.arange(self.t.shape[1], dtype=np.int64) + np.max(t2f) + 1

        doflocs = np.hstack((
            p,
            .5 * np.sum(p[:, self.edges], axis=1),
            .25 * np.sum(p[:, self.facets], axis=1),
            .125 * np.sum(p[:, t], axis=1),
        ))
        t = np.hstack((
            np.vstack((t[0], t2e[0], t2e[1], t2e[2],
                       t2f[0], t2f[2], t2f[1], mid)),
            np.vstack((t2e[0], t[1], t2f[0], t2f[2],
                       t2e[3], t2e[4], mid, t2f[4])),
            np.vstack((t2e[1], t2f[0], t[2], t2f[1],
                       t2e[5], mid, t2e[6], t2f[3])),
            np.vstack((t2e[2], t2f[2], t2f[1], t[3],
                       mid, t2e[7], t2e[8], t2f[5])),
            np.vstack((t2f[0], t2e[3], t2e[5], mid,
                       t[4], t2f[4], t2f[3], t2e[9])),
            np.vstack((t2f[2], t2e[4], mid, t2e[7],
                       t2f[4], t[5], t2f[5], t2e[10])),
            np.vstack((t2f[1], mid, t2e[6], t2e[8],
                       t2f[3], t2f[5], t[6], t2e[11])),
            np.vstack((mid, t2f[4], t2f[3], t2f[5],
                       t2e[9], t2e[10], t2e[11], t[7]))
        ))
        return replace(
            self,
            doflocs=doflocs,
            t=t,
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
        return cls(p, t.astype(np.int64))

    def to_meshtet(self):
        """Split each hexahedron into six tetrahedra."""
        t = np.hstack((
            self.t[[0, 1, 3, 4]],
            self.t[[0, 3, 2, 4]],
            self.t[[2, 3, 4, 6]],
            self.t[[3, 4, 6, 7]],
            self.t[[3, 4, 5, 7]],
            self.t[[1, 3, 4, 5]],
        ))

        return MeshTet1(self.doflocs, t)

    def element_finder(self, mapping=None):
        """Transform to :class:`skfem.MeshTet` and return its finder."""
        tet_finder = self.to_meshtet().element_finder()

        def finder(*args):
            return tet_finder(*args) % self.t.shape[1]

        return finder
