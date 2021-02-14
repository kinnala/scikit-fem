from dataclasses import dataclass, replace
from typing import Any, Optional

from numpy import ndarray

from .graph import Graph
from ..assembly import Dofs
from ..element import Element, ElementVector


@dataclass
class Geometry(Graph):

    elem: Element
    doflocs: ndarray
    affine: bool = False

    @property
    def p(self):
        return self.doflocs

    @property
    def dofs(self):
        if not hasattr(self, '_dofs'):
            self._dofs = Dofs(self, self.elem)
        return self._dofs
    
    @property
    def refdom(self):  # todo
        return self.elem.mesh_type.refdom

    @property
    def brefdom(self):  # todo
        return self.elem.mesh_type.brefdom

    def _mapping(self):
        from skfem.mapping import MappingAffine, MappingIsoparametric
        #if self.doflocs is None:
        #    return MappingAffine(self)
        from typing import NamedTuple

        class FakeMesh(NamedTuple):
            p: ndarray
            t: ndarray
            facets: ndarray

        fakemesh = FakeMesh(
            self.doflocs,
            self.dofs.element_dofs,
            self.facets,
        )

        if self.affine:
            return MappingAffine(fakemesh)

        return MappingIsoparametric(
            fakemesh,
            self.elem,
            self.bndelem,
        )

    @property
    def bndelem(self):

        from skfem.element import (ElementHex1, ElementHex2, ElementLineP1,
                                   ElementLineP2, ElementQuad1, ElementQuad2,
                                   ElementTetP1, ElementTetP2, ElementTriP1,
                                   ElementTriP2)

        BOUNDARY_ELEMENT_MAP = {
            ElementTriP1: ElementLineP1,
            ElementTriP2: ElementLineP2,
            ElementQuad1: ElementLineP1,
            ElementQuad2: ElementLineP2,
            ElementTetP1: ElementTriP1,
            ElementTetP2: ElementTriP2,
            ElementHex1: ElementQuad1,
            ElementHex2: ElementQuad2,
        }

        return BOUNDARY_ELEMENT_MAP[type(self.elem)]()
