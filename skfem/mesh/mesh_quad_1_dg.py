from dataclasses import dataclass
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementQuad1DG
from .mesh_quad_1 import MeshQuad1


@dataclass(repr=False)
class MeshQuad1DG(MeshQuad1):
    """A quadrilateral mesh with a discontinuous topology."""
    elem: Type[Element] = ElementQuad1DG
