from dataclasses import dataclass, replace
from typing import Any, Optional

from numpy import ndarray

from .graph import Graph


@dataclass
class Geometry(Graph):

    elem: Any  # this is Element but cannot import it yet
    doflocs: Optional[ndarray] = None
    dofs: Optional[ndarray] = None

    @property
    def _p(self):
        return self.p if self.doflocs is None else self.doflocs

    @property
    def _t(self):
        return self.t if self.dofs is None else self.dofs

    @property
    def refdom(self):  # todo
        return self.elem.mesh_type.refdom

    @property
    def brefdom(self):  # todo
        return self.elem.mesh_type.brefdom

    def _mapping(self):
        from skfem.mapping import MappingAffine, MappingIsoparametric
        if self.doflocs is None:
            return MappingAffine(self)
        return MappingIsoparametric(
            replace(self, p=self._p, t=self._t),
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
