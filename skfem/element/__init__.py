"""The module :mod:`skfem.element` defines finite elements in a very generic
sense.

In order to use an element, you simply initialize the respective object and
pass it to the constructor of :class:`~skfem.assembly.CellBasis` or
:class:`~skfem.assembly.FacetBasis`.  See below for a list of supported
elements.

"""
from typing import Type, Dict
from skfem.generic_utils import Removed
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
                          ElementTriRT2, ElementTriN1, ElementTriN2,
                          ElementTriN3, ElementTriHHJ0, ElementTriHHJ1,
                          ElementTriBubble, ElementTriP,
                          ElementTriRT)  # noqa
from .element_quad import (ElementQuad0, ElementQuad1, ElementQuad2,  # noqa
                           ElementQuadS2, ElementQuadP,
                           ElementQuadBFS, ElementQuadRT1, ElementQuad1DG,
                           ElementQuadN1, ElementQuad2G, ElementQuadDPC)
from .element_tet import (ElementTetP0, ElementTetP1, ElementTetP2,  # noqa
                          ElementTetRT1, ElementTetN1, ElementTetMini,
                          ElementTetCR, ElementTetCCR, ElementTetSkeletonP0,
                          ElementTetBubble, ElementTetMorley,
                          ElementTetMWX3)
from .element_hex import (ElementHex0, ElementHex1, ElementHex2,  # noqa
                          ElementHexS2, ElementHex1DG, ElementHexRT1,
                          ElementHexSkeleton0, ElementHexC1, ElementHexDPC)
from .element_line import (ElementLineP0, ElementLineP1, ElementLineP2,  # noqa
                           ElementLinePp, ElementLineHermite,
                           ElementLineMini, ElementLineP1DG,
                           ElementLineBubble)
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
    ElementHex2: ElementQuad2,
    ElementHex1DG: ElementQuad1,
}

# removed aliases, kept importable for their migration messages
ElementVectorH1 = Removed(
    "skfem.ElementVectorH1", version="13.0.0",
    message="Use ElementVector instead of ElementVectorH1.")
ElementTriDG = Removed(
    "skfem.ElementTriDG", version="13.0.0",
    message="Use ElementDG instead of ElementTriDG.")
ElementQuadDG = Removed(
    "skfem.ElementQuadDG", version="13.0.0",
    message="Use ElementDG instead of ElementQuadDG.")
ElementHexDG = Removed(
    "skfem.ElementHexDG", version="13.0.0",
    message="Use ElementDG instead of ElementHexDG.")
ElementTetDG = Removed(
    "skfem.ElementTetDG", version="13.0.0",
    message="Use ElementDG instead of ElementTetDG.")
ElementTriMini = Removed(
    "skfem.ElementTriMini", version="13.0.0",
    message="Use ElementTriP1B instead of ElementTriMini.")
ElementTriCCR = Removed(
    "skfem.ElementTriCCR", version="13.0.0",
    message="Use ElementTriP2B instead of ElementTriCCR.")
ElementTriRT0 = Removed(
    "skfem.ElementTriRT0", version="13.0.0",
    message="Use ElementTriRT1 instead of ElementTriRT0.")
ElementTetRT0 = Removed(
    "skfem.ElementTetRT0", version="13.0.0",
    message="Use ElementTetRT1 instead of ElementTetRT0.")
ElementQuadRT0 = Removed(
    "skfem.ElementQuadRT0", version="13.0.0",
    message="Use ElementQuadRT1 instead of ElementQuadRT0.")
ElementTetN0 = Removed(
    "skfem.ElementTetN0", version="13.0.0",
    message="Use ElementTetN1 instead of ElementTetN0.")


__all__ = [
    "DiscreteField",
    "Element",
    "ElementH1",
    "ElementVector",
    "ElementHdiv",
    "ElementHcurl",
    "ElementGlobal",
    "ElementDG",
    "ElementComposite",
    "ElementTriP1",
    "ElementTriP2",
    "ElementTriP3",
    "ElementTriP4",
    "ElementTriP0",
    "ElementTriCR",
    "ElementTriRT1",
    "ElementTriRT2",
    "ElementTriMorley",
    "ElementTri15ParamPlate",
    "ElementTriArgyris",
    "ElementTriHermite",
    "ElementTriP1DG",
    "ElementTriSkeletonP0",
    "ElementTriSkeletonP1",
    "ElementTriBDM1",
    "ElementTriP1G",
    "ElementTriP2G",
    "ElementTriP1B",
    "ElementTriP2B",
    "ElementTriN1",
    "ElementTriN2",
    "ElementTriN3",
    "ElementTriHHJ0",
    "ElementTriHHJ1",
    "ElementTriBubble",
    "ElementTriP",
    "ElementTriRT",
    "ElementQuad0",
    "ElementQuad1",
    "ElementQuad2",
    "ElementQuadS2",
    "ElementQuadP",
    "ElementQuadBFS",
    "ElementQuadRT1",
    "ElementQuad1DG",
    "ElementQuadN1",
    "ElementQuad2G",
    "ElementQuadDPC",
    "ElementTetP0",
    "ElementTetP1",
    "ElementTetP2",
    "ElementTetRT1",
    "ElementTetN1",
    "ElementTetMini",
    "ElementTetCR",
    "ElementTetCCR",
    "ElementTetSkeletonP0",
    "ElementTetBubble",
    "ElementTetMorley",
    "ElementTetMWX3",
    "ElementHex0",
    "ElementHex1",
    "ElementHex2",
    "ElementHexS2",
    "ElementHex1DG",
    "ElementHexRT1",
    "ElementHexSkeleton0",
    "ElementHexDPC",
    "ElementLineP0",
    "ElementLineP1",
    "ElementLineP1DG",
    "ElementLineP2",
    "ElementLinePp",
    "ElementLineHermite",
    "ElementLineMini",
    "ElementLineBubble",
    "ElementWedge1",
    "ElementHexC1",
]
