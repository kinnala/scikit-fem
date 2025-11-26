from dataclasses import dataclass
from typing import Type

from ..element import Element, ElementHex1DG
from .mesh_hex_1 import MeshHex1
from .mesh_dg import MeshDG


@dataclass(repr=False)
class MeshHex1DG(MeshDG, MeshHex1):
    """A hexahedral mesh with a discontinuous topology."""
    elem: Type[Element] = ElementHex1DG
