from dataclasses import dataclass
from typing import Type

from ..element import Element, ElementLineP1DG
from .mesh_line_1 import MeshLine1
from .mesh_dg import MeshDG


@dataclass(repr=False)
class MeshLine1DG(MeshDG, MeshLine1):
    """One-dimensional mesh with a discontinuous topology."""
    elem: Type[Element] = ElementLineP1DG
    affine: bool = False
    sort_t: bool = False
