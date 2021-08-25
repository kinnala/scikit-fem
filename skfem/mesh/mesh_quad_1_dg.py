from dataclasses import dataclass
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementQuad1DG
from .mesh_quad_1 import MeshQuad1
from .mesh_dg import MeshDG


@dataclass(repr=False)
class MeshQuad1DG(MeshQuad1, MeshDG):
    """A quadrilateral mesh with a discontinuous topology."""
    elem: Type[Element] = ElementQuad1DG
