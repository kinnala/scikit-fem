from dataclasses import dataclass, field
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementHex2
from .mesh_hex_1 import MeshHex1


@dataclass(repr=False)
class MeshHex2(MeshHex1):

    doflocs: ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 1.0],
                [0.5, 0.0, 1.0],
                [0.0, 1.0, 0.5],
                [0.5, 1.0, 0.0],
                [1.0, 0.0, 0.5],
                [1.0, 0.5, 0.0],
                [0.5, 1.0, 1.0],
                [1.0, 0.5, 1.0],
                [1.0, 1.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 1.0, 0.5],
                [1.0, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        ).T
    )
    elem: Type[Element] = ElementHex2

    def _uniform(self):
        return MeshHex2.from_mesh(MeshHex1.from_mesh(self).refined())
