from dataclasses import dataclass
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementTriP1DG
from .mesh_tri_1 import MeshTri1
from .mesh_dg import MeshDG


@dataclass(repr=False)
class MeshTri1DG(MeshDG, MeshTri1):
    """A triangular mesh with a discontinuous topology."""
    doflocs: ndarray = np.array([[0., 0.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 1.]], dtype=np.float64).T
    elem: Type[Element] = ElementTriP1DG
    affine: bool = False
    sort_t: bool = False
