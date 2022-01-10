"""The module :mod:`skfem.element` defines finite elements in a very generic
sense.

In order to use an element, you simply initialize the respective object and
pass it to the constructor of :class:`~skfem.assembly.CellBasis` or
:class:`~skfem.assembly.BoundaryFacetBasis`.  See below for a list of supported
elements.

"""
from typing import Type, Dict
from .discrete_field import DiscreteField  # noqa
from .element import Element
from .element_h1 import ElementH1  # noqa
from .element_hdiv import ElementHdiv  # noqa
from .element_hcurl import ElementHcurl  # noqa
from .element_global import ElementGlobal  # noqa
from .element_vector import ElementVector
from .element_tri import (ElementTriP1, ElementTriP2,  # noqa
                          ElementTriP0, ElementTriRT0, ElementTriMorley,
                          ElementTriArgyris, ElementTriMini, ElementTriCR,
                          ElementTriHermite, ElementTriCCR,
                          ElementTriP1DG, ElementTriSkeletonP0,
                          ElementTriSkeletonP1, ElementTriP3,
                          ElementTriP4, ElementTri15ParamPlate,
                          ElementTriBDM1)
from .element_quad import (ElementQuad0, ElementQuad1, ElementQuad2,  # noqa
                           ElementQuadS2, ElementQuadP,
                           ElementQuadBFS, ElementQuadRT0, ElementQuad1DG)
from .element_tet import (ElementTetP0, ElementTetP1, ElementTetP2,  # noqa
                          ElementTetRT0, ElementTetN0, ElementTetMini,
                          ElementTetCR, ElementTetCCR)
from .element_hex import (ElementHex0, ElementHex1, ElementHex2,  # noqa
                          ElementHexS2, ElementHex1DG)
from .element_line import (ElementLineP0, ElementLineP1, ElementLineP2,  # noqa
                           ElementLinePp, ElementLineHermite,
                           ElementLineMini, ElementLineP1DG)
from .element_wedge_1 import ElementWedge1  # noqa
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
