from dataclasses import dataclass
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementWedge1
from .mesh_3d import Mesh3D
from .mesh_tet_1 import MeshTet1


@dataclass(repr=False)
class MeshWedge1(Mesh3D):

    doflocs: ndarray = np.array([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 1., 1.],
                                 [1., 0., 1.],
                                 [1., 1., 0.],
                                 [1., 1., 1.]], dtype=np.float64).T
    t: ndarray = np.array([[0, 2, 3, 1, 4, 5],
                           [2, 3, 6, 4, 5, 7]], dtype=np.int64).T
    elem: Type[Element] = ElementWedge1

    def to_meshtet(self):

        t = np.hstack((
            self.t[[0, 1, 2, 3]],
            self.t[[1, 2, 3, 4]],
            self.t[[2, 3, 4, 5]],
        ))

        return MeshTet1(self.doflocs, t)

    def element_finder(self, mapping=None):
        """Transform to :class:`skfem.MeshTet` and return its finder."""
        tet_finder = self.to_meshtet().element_finder()

        def finder(*args):
            return tet_finder(*args) % self.t.shape[1]

        return finder
