from dataclasses import dataclass, replace, field
from itertools import dropwhile
from typing import Optional, Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementHexagon1
from .mesh_2d import Mesh2D
from .mesh_tri_1 import MeshTri1


@dataclass(repr=False)
class MeshHexagon1(Mesh2D):
    """Hexagonal mesh."""

    doflocs: ndarray = field(
        default_factory=lambda: np.array(
            [[1.0, 0.0],
             [0.5, np.sqrt(3) / 2],
             [-0.5, np.sqrt(3) / 2],
             [-1.0, 0.0],
             [-0.5, -np.sqrt(3) / 2],
             [0.5, -np.sqrt(3) / 2]], dtype=np.float64
        ).T
    )
    t: ndarray = field(
        default_factory=lambda: np.array([[0, 1, 2, 3, 4, 5]],
                                         dtype=np.int64).T
    )
    elem: Type[Element] = ElementHexagon1

    def to_meshtri(self, x: Optional[ndarray] = None):
        """Split each hexagon into six triangles."""

        tnew = np.arange(np.max(self.t) + 1,
                         np.max(self.t) + 1 + self.t.shape[1],
                         dtype=np.int64)
        t = np.hstack((
            np.vstack((self.t[[0, 1]], tnew)),
            np.vstack((self.t[[1, 2]], tnew)),
            np.vstack((self.t[[2, 3]], tnew)),
            np.vstack((self.t[[3, 4]], tnew)),
            np.vstack((self.t[[4, 5]], tnew)),
            np.vstack((self.t[[5, 0]], tnew)),
        ))

        nt = self.t.shape[1]

        p = np.hstack((self.doflocs,
                       self.doflocs[:, self.t].mean(axis=1)))

        mesh = MeshTri1(p, t)

        if x is not None:
            X = np.concatenate((x, x, x, x, x, x))
            return mesh, X

        return mesh


    def _uniform(self):
        return self.to_meshtri()._uniform()


    def element_finder(self, mapping=None):
        """Transform to :class:`skfem.MeshTri` and return its finder."""
        tri_finder = self.to_meshtri().element_finder()

        def finder(*args):
            return tri_finder(*args) % self.t.shape[1]

        return finder
