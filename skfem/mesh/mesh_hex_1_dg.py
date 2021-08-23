from dataclasses import dataclass
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementHex1DG
from .mesh_hex_1 import MeshHex1


@dataclass(repr=False)
class MeshHex1DG(MeshHex1):
    """A hexahedral mesh with a discontinuous topology."""
    doflocs: ndarray = np.array([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 1., 1.],
                                 [1., 0., 1.],
                                 [1., 1., 0.],
                                 [1., 1., 1.]], dtype=np.float64).T
    elem: Type[Element] = ElementHex1DG
