from dataclasses import dataclass
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementQuad1DG
from .mesh_quad_1 import MeshQuad1


@dataclass(repr=False)
class MeshQuad1DG(MeshQuad1):
    """A quadrilateral mesh with a discontinuous topology."""
    doflocs: ndarray = np.array([[0., 0.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 1.]], dtype=np.float64).T
    elem: Type[Element] = ElementQuad1DG
