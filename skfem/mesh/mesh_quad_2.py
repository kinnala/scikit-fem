from dataclasses import dataclass
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementQuad2
from .mesh_quad_1 import MeshQuad1
from .mesh_2d_2 import Mesh2D2


@dataclass(repr=False)
class MeshQuad2(Mesh2D2, MeshQuad1):
    """A quadratic quadrilateral mesh."""

    doflocs: ndarray = np.array([[0., 0.],
                                 [1., 0.],
                                 [1., 1.],
                                 [0., 1.],
                                 [.5, 0.],
                                 [0., .5],
                                 [1., .5],
                                 [.5, 1.],
                                 [.5, .5]], dtype=np.float64).T
    elem: Type[Element] = ElementQuad2
