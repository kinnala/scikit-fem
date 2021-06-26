from dataclasses import dataclass
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementQuad2
from .mesh_quad_1 import MeshQuad1


@dataclass(repr=False)
class MeshQuad2(MeshQuad1):
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

    def _repr_svg_(self) -> str:
        from skfem.visuals.svg import draw
        return draw(self, nrefs=2, boundaries_only=True).svg
