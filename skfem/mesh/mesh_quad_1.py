from dataclasses import dataclass, replace
from itertools import dropwhile
from typing import Optional, Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementQuad1
from .mesh_2d import Mesh2D
from .mesh_tri_1 import MeshTri1


@dataclass(repr=False)
class MeshQuad1(Mesh2D):
    """A standard first-order quadrilateral mesh.

    If ``t`` is provided, order of vertices in each element should match the
    numbering::

          3---2
          |   |
          0---1

    """
    doflocs: ndarray = np.array([[0., 0.],
                                 [1., 0.],
                                 [1., 1.],
                                 [0., 1.]], dtype=np.float64).T
    t: ndarray = np.array([[0, 1, 2, 3]], dtype=np.int64).T
    elem: Type[Element] = ElementQuad1

    def _uniform(self):

        p = self.doflocs
        t = self.t
        sz = p.shape[1]
        t2f = self.t2f.copy() + sz
        mid = np.arange(t.shape[1], dtype=np.int64) + np.max(t2f) + 1
        return replace(
            self,
            doflocs=np.hstack((
                p,
                p[:, self.facets].mean(axis=1),
                p[:, self.t].mean(axis=1),
            )),
            t=np.hstack((
                np.vstack((t[0], t2f[0], mid, t2f[3])),
                np.vstack((t2f[0], t[1], t2f[1], mid)),
                np.vstack((mid, t2f[1], t[2], t2f[2])),
                np.vstack((t2f[3], mid, t2f[2], t[3])),
            )),
            _boundaries=None,
            _subdomains=None,
        )

    @classmethod
    def init_tensor(cls: Type,
                    x: ndarray,
                    y: ndarray):
        """Initialize a tensor product mesh.

        The mesh topology is as follows::

            *-------------*
            |   |  |      |
            |---+--+------|
            |   |  |      |
            |   |  |      |
            |   |  |      |
            *-------------*

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
        t = np.zeros((4, nt))
        ix = ix.reshape(npy, npx, order='F').copy()
        t[0] = (ix[0:(npy-1), 0:(npx-1)].reshape(nt, 1, order='F')
                .copy()
                .flatten())
        t[1] = (ix[1:npy, 0:(npx-1)].reshape(nt, 1, order='F')
                .copy()
                .flatten())
        t[2] = (ix[1:npy, 1:npx].reshape(nt, 1, order='F')
                .copy()
                .flatten())
        t[3] = (ix[0:(npy-1), 1:npx].reshape(nt, 1, order='F')
                .copy()
                .flatten())
        return cls(p, t.astype(np.int64))

    def to_meshtri(self, x: Optional[ndarray] = None):
        """Split each quadrilateral into two triangles."""
        t = np.hstack((self.t[[0, 1, 3]], self.t[[1, 2, 3]]))

        subdomains = None
        if self.subdomains:
            subdomains = {k: np.concatenate((v, v + self.t.shape[1]))
                          for k, v in self.subdomains.items()}

        mesh = MeshTri1(self.doflocs, t)

        boundaries = None
        if self.boundaries:
            boundaries = {}
            for k in self.boundaries:
                slots = enumerate(mesh.facets.T)
                boundaries[k] = np.array([
                    next(dropwhile(lambda slot: not(np.array_equal(f,
                                                                   slot[1])),
                                   slots))[0]
                    for f in self.facets.T[np.sort(self.boundaries[k])]])

        if self._subdomains or self._boundaries:
            mesh = replace(
                mesh,
                _boundaries=boundaries,
                _subdomains=subdomains,
            )

        if x is not None:
            if len(x) == self.t.shape[1]:
                # preserve elemental constant functions
                X = np.concatenate((x, x))
            else:
                raise Exception("The parameter x must have one value per "
                                "element.")
            return mesh, X
        return mesh

    def element_finder(self, mapping=None):
        """Transform to :class:`skfem.MeshTri` and return its finder."""
        tri_finder = self.to_meshtri().element_finder()

        def finder(*args):
            return tri_finder(*args) % self.t.shape[1]

        return finder
