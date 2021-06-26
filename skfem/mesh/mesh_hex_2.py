from dataclasses import dataclass
from typing import Type

from ..element import Element, ElementHex2
from .mesh_hex_1 import MeshHex1


@dataclass(repr=False)
class MeshHex2(MeshHex1):

    elem: Type[Element] = ElementHex2
