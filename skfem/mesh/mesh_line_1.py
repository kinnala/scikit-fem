from dataclasses import dataclass, replace
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementLineP1
from .mesh import Mesh
from .mesh_quad_1 import MeshQuad1


@dataclass(repr=False)
class MeshLine1(Mesh):
    """A one-dimensional mesh."""

    doflocs: ndarray = np.array([[0., 1.]], dtype=np.float64)
    t: ndarray = np.array([[0], [1]], dtype=np.int64)
    elem: Type[Element] = ElementLineP1
    affine: bool = True

    def __mul__(self, other):

        from .mesh_line_1 import MeshLine1

        if isinstance(other, MeshLine1):
            return MeshQuad1.init_tensor(self.p[0], other.p[0])

        return other * self

    def _uniform(self):
        p, t = self.doflocs, self.t

        newp = np.hstack((p, p[:, t].mean(axis=1)))
        newt = np.empty((t.shape[0], 2 * t.shape[1]),
                        dtype=t.dtype)
        newt[0, ::2] = t[0]
        newt[0, 1::2] = p.shape[1] + np.arange(t.shape[1])
        newt[1, ::2] = newt[0, 1::2]
        newt[1, 1::2] = t[1]

        return replace(
            self,
            doflocs=newp,
            t=newt,
            _boundaries=None,
            _subdomains=None,
        )

    def _adaptive(self, marked):
        p, t = self.doflocs, self.t

        mid = range(len(marked)) + np.max(t) + 1
        nonmarked = np.setdiff1d(np.arange(t.shape[1]), marked)
        newp = np.hstack((p, p[:, t[:, marked]].mean(1)))
        newt = np.vstack((t[0, marked], mid))
        newt = np.hstack((t[:, nonmarked],
                          newt,
                          np.vstack((mid, t[1, marked]))))

        return replace(
            self,
            doflocs=newp,
            t=newt,
        )

    def param(self):
        return np.max(np.abs(self.p[0, self.t[1]] - self.p[0, self.t[0]]))

    def element_finder(self, mapping=None):

        ix = np.argsort(self.p[0])
        maxt = self.t[np.argmax(self.p[0, self.t], 0),
                      np.arange(self.t.shape[1])]

        def finder(x):
            xin = x.copy()  # bring endpoint inside for np.digitize
            xin[x == self.p[0, ix[-1]]] = self.p[0, ix[-2:]].mean()
            elems = np.nonzero(ix[np.digitize(xin, self.p[0, ix])][:, None]
                               == maxt)[1]
            if len(elems) < len(x):
                raise ValueError("Point is outside of the mesh.")
            return elems

        return finder

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p[:, :1]
