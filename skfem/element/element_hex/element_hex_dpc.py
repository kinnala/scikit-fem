from ..element_quad.element_quad_dpc import ElementQuadDPC
from ...refdom import RefHex


class ElementHexDPC(ElementQuadDPC):
    """Discontinuous element with polynomials of total degree :math:`p`.

    See :class:`skfem.element.ElementQuadDPC` for more information.

    """

    refdom = RefHex
