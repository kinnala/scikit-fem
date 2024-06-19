from dataclasses import dataclass, replace, field
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

    doflocs: ndarray = field(
        default_factory=lambda: np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
        ).T
    )
    t: ndarray = field(
        default_factory=lambda: np.array([[0, 1, 2, 3]], dtype=np.int32).T
    )
    elem: Type[Element] = ElementQuad1

    def _uniform(self):

        p = self.doflocs
        t = self.t
        sz = p.shape[1]
        t2f = self.t2f.copy()
        mid = np.arange(t.shape[1], dtype=np.int32) + np.max(t2f) + sz + 1

        m = replace(
            self,
            doflocs=np.hstack((
                p,
                p[:, self.facets].mean(axis=1),
                p[:, self.t].mean(axis=1),
            )),
            t=np.hstack((
                np.vstack((t[0], t2f[0] + sz, mid, t2f[3] + sz)),
                np.vstack((t2f[0] + sz, t[1], t2f[1] + sz, mid)),
                np.vstack((mid, t2f[1] + sz, t[2], t2f[2] + sz)),
                np.vstack((t2f[3] + sz, mid, t2f[2] + sz, t[3])),
            )),
            _boundaries=None,
            _subdomains=None,
        )

        if self._boundaries is not None:
            # mapping of indices between old and new facets
            new_facets = np.zeros((2, self.facets.shape[1]), dtype=np.int32)
            ix0 = np.arange(t.shape[1], dtype=np.int32)
            ix1 = ix0 + t.shape[1]
            ix2 = ix0 + 2 * t.shape[1]
            ix3 = ix0 + 3 * t.shape[1]

            # finish mapping of indices between old and new facets
            new_facets[0, t2f[0]] = m.t2f[0, ix0]
            new_facets[1, t2f[0]] = m.t2f[0, ix1]
            new_facets[0, t2f[1]] = m.t2f[1, ix1]
            new_facets[1, t2f[1]] = m.t2f[1, ix2]
            new_facets[0, t2f[2]] = m.t2f[2, ix2]
            new_facets[1, t2f[2]] = m.t2f[2, ix3]
            new_facets[0, t2f[3]] = m.t2f[3, ix3]
            new_facets[1, t2f[3]] = m.t2f[3, ix0]

            m = replace(
                m,
                _boundaries={
                    name: np.sort(new_facets[:, ixs].flatten())
                    for name, ixs in self._boundaries.items()
                },
            )

        return m

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
        return cls(p, t.astype(np.int32))

    def to_meshtri(self,
                   x: Optional[ndarray] = None,
                   style: Optional[str] = None):
        """Split each quadrilateral into two triangles.

        Parameters
        ----------
        x
            Elementwise constant function to preserve. If given, returned as
            the second additional output parameter.
        style
            Optionally, specify a splitting style `'x'` for crisscross
            splitting.

        """
        if style == 'x':
            tnew = np.arange(np.max(self.t) + 1,
                             np.max(self.t) + 1 + self.t.shape[1],
                             dtype=np.int32)
            t = np.hstack((
                np.vstack((self.t[[0, 1]], tnew)),
                np.vstack((self.t[[1, 2]], tnew)),
                np.vstack((self.t[[2, 3]], tnew)),
                np.vstack((self.t[[0, 3]], tnew)),
            ))
        else:
            t = np.hstack((self.t[[0, 1, 3]], self.t[[1, 2, 3]]))

        nt = self.t.shape[1]

        subdomains = None
        if self.subdomains:
            if style == 'x':
                subdomains = {k: np.concatenate((v,
                                                 v + nt,
                                                 v + 2 * nt,
                                                 v + 3 * nt))
                              for k, v in self.subdomains.items()}
            else:
                subdomains = {k: np.concatenate((v, v + nt))
                              for k, v in self.subdomains.items()}

        if style == 'x':
            p = np.hstack((self.doflocs,
                           self.doflocs[:, self.t].mean(axis=1)))
        else:
            p = self.doflocs

        mesh = MeshTri1(p, t)

        boundaries = None
        if self.boundaries:
            boundaries = {}
            for k in self.boundaries:
                slots = enumerate(mesh.facets.T)
                boundaries[k] = np.array([
                    next(dropwhile(lambda s: (not np.array_equal(f, s[1])),
                                   slots))[0]
                    for f in self.facets.T[np.sort(self.boundaries[k])]])

        if self._subdomains or self._boundaries:
            mesh = replace(
                mesh,
                _boundaries=boundaries,
                _subdomains=subdomains,
            )

        if x is not None:
            if len(x) == nt:
                # preserve elemental constant functions
                if style == 'x':
                    X = np.concatenate((x, x, x, x))
                else:
                    X = np.concatenate((x, x))
            else:
                raise NotImplementedError("The parameter x must have one "
                                          "value per element.")
            return mesh, X
        return mesh

    def element_finder(self, mapping=None):
        """Transform to :class:`skfem.MeshTri` and return its finder."""
        tri_finder = self.to_meshtri().element_finder()

        def finder(*args):
            return tri_finder(*args) % self.t.shape[1]

        return finder
