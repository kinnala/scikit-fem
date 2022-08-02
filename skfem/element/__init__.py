"""The module :mod:`skfem.element` defines finite elements in a very generic
sense.

In order to use an element, you simply initialize the respective object and
pass it to the constructor of :class:`~skfem.assembly.CellBasis` or
:class:`~skfem.assembly.FacetBasis`.  See below for a list of supported
elements.

"""
from typing import Type, Dict
from .discrete_field import DiscreteField
from .element import Element
from .element_h1 import ElementH1
from .element_hdiv import ElementHdiv
from .element_hcurl import ElementHcurl
from .element_global import ElementGlobal
from .element_vector import ElementVector
from .element_tri import (ElementTriP1, ElementTriP2, ElementTriP0,
                          ElementTriRT1, ElementTriMorley, ElementTriArgyris,
                          ElementTriP1B, ElementTriCR, ElementTriHermite,
                          ElementTriP1DG, ElementTriSkeletonP0,
                          ElementTriSkeletonP1, ElementTriP3, ElementTriP4,
                          ElementTri15ParamPlate, ElementTriBDM1,
                          ElementTriP1G, ElementTriP2G, ElementTriP2B,
                          ElementTriRT2)  # noqa
from .element_quad import (ElementQuad0, ElementQuad1, ElementQuad2,  # noqa
                           ElementQuadS2, ElementQuadP,
                           ElementQuadBFS, ElementQuadRT1, ElementQuad1DG)
from .element_tet import (ElementTetP0, ElementTetP1, ElementTetP2,  # noqa
                          ElementTetRT1, ElementTetN0, ElementTetMini,
                          ElementTetCR, ElementTetCCR)
from .element_hex import (ElementHex0, ElementHex1, ElementHex2,  # noqa
                          ElementHexS2, ElementHex1DG, ElementHexRT1)
from .element_line import (ElementLineP0, ElementLineP1, ElementLineP2,  # noqa
                           ElementLinePp, ElementLineHermite,
                           ElementLineMini, ElementLineP1DG)
from .element_wedge_1 import ElementWedge1
from .element_composite import ElementComposite  # noqa
from .element_dg import ElementDG


BOUNDARY_ELEMENT_MAP: Dict[Type[Element], Type[Element]] = {
    ElementTriP0: ElementLineP0,
    ElementTriP1: ElementLineP1,
    ElementTriP2: ElementLineP2,
    ElementQuad0: ElementLineP0,
    ElementQuad1: ElementLineP1,
    ElementQuad2: ElementLineP2,
    ElementTetP0: ElementTriP0,
    ElementTetP1: ElementTriP1,
    ElementTetP2: ElementTriP2,
    ElementHex0: ElementQuad0,
    ElementHex1: ElementQuad1,
    ElementHex2: ElementQuad2
}

# for backwards compatibility
ElementVectorH1 = ElementVector
ElementTriDG = ElementDG
ElementQuadDG = ElementDG
ElementHexDG = ElementDG
ElementTetDG = ElementDG
ElementTriMini = ElementTriP1B
ElementTriCCR = ElementTriP2B
ElementTriRT0 = ElementTriRT1
ElementTetRT0 = ElementTetRT1
ElementQuadRT0 = ElementQuadRT1


__all__ = [
    "DiscreteField",
    "Element",
    "ElementH1",
    "ElementVector",
    "ElementVectorH1",
    "ElementHdiv",
    "ElementHcurl",
    "ElementGlobal",
    "ElementDG",
    "ElementComposite",
    "ElementTriP1",
    "ElementTriP2",
    "ElementTriP3",
    "ElementTriP4",
    "ElementTriDG",
    "ElementTriP0",
    "ElementTriCR",
    "ElementTriCCR",
    "ElementTriRT0",
    "ElementTriRT1",
    "ElementTriRT2",
    "ElementTriMorley",
    "ElementTri15ParamPlate",
    "ElementTriArgyris",
    "ElementTriMini",
    "ElementTriHermite",
    "ElementTriP1DG",
    "ElementTriSkeletonP0",
    "ElementTriSkeletonP1",
    "ElementTriBDM1",
    "ElementTriP1G",
    "ElementTriP2G",
    "ElementTriP1B",
    "ElementTriP2B",
    "ElementQuad0",
    "ElementQuad1",
    "ElementQuad2",
    "ElementQuadS2",
    "ElementQuadDG",
    "ElementQuadP",
    "ElementQuadBFS",
    "ElementQuadRT0",
    "ElementQuadRT1",
    "ElementQuad1DG",
    "ElementTetP0",
    "ElementTetP1",
    "ElementTetP2",
    "ElementTetRT0",
    "ElementTetRT1",
    "ElementTetN0",
    "ElementTetMini",
    "ElementTetCR",
    "ElementTetCCR",
    "ElementTetDG",
    "ElementHex0",
    "ElementHex1",
    "ElementHex2",
    "ElementHexS2",
    "ElementHex1DG",
    "ElementHexDG",
    "ElementHexRT1",
    "ElementLineP0",
    "ElementLineP1",
    "ElementLineP1DG",
    "ElementLineP2",
    "ElementLinePp",
    "ElementLineHermite",
    "ElementLineMini",
    "ElementWedge1",
]
