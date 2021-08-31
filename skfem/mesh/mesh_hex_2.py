from dataclasses import dataclass
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementHex2
from .mesh_hex_1 import MeshHex1


@dataclass(repr=False)
class MeshHex2(MeshHex1):

    doflocs: ndarray = np.array([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 1., 1.],
                                 [1., 0., 1.],
                                 [1., 1., 0.],
                                 [1., 1., 1.],
                                 [0., 0., .5],
                                 [0., .5, 0.],
                                 [.5, 0., 0.],
                                 [0., .5, 1.],
                                 [.5, 0., 1.],
                                 [0., 1., .5],
                                 [.5, 1., 0.],
                                 [1., 0., .5],
                                 [1., .5, 0.],
                                 [.5, 1., 1.],
                                 [1., .5, 1.],
                                 [1., 1., .5],
                                 [0., .5, .5],
                                 [.5, 0., .5],
                                 [.5, .5, 0.],
                                 [.5, .5, 1.],
                                 [.5, 1., .5],
                                 [1., .5, .5],
                                 [.5, .5, .5]], dtype=np.float64).T
    elem: Type[Element] = ElementHex2
