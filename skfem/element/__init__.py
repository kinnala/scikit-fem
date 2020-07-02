"""This module defines finite elements (in a very generic sense).

The naming of the element classes reflects their compatibility with the mesh
types.

>>> from skfem.element import ElementTriP1
>>> ElementTriP1.mesh_type
skfem.mesh.mesh2d.mesh_tri.MeshTri

In order to use an element, you simply initialize the respective object and
pass it to the constructor of :class:`~skfem.assembly.InteriorBasis` or
:class:`~skfem.assembly.FacetBasis`.

The supported elements include

    - :class:`~skfem.element.ElementTriArgyris`
    - :class:`~skfem.element.ElementHex1`
    - :class:`~skfem.element.ElementTriMorley`
    - :class:`~skfem.element.ElementTriMini`
    - :class:`~skfem.element.ElementQuad0`
    - :class:`~skfem.element.ElementQuad1`
    - :class:`~skfem.element.ElementQuad2`
    - :class:`~skfem.element.ElementQuadS2`
    - :class:`~skfem.element.ElementQuadDG`
    - :class:`~skfem.element.ElementQuadP`
    - :class:`~skfem.element.ElementQuadBFS`
    - :class:`~skfem.element.ElementTetN0`
    - :class:`~skfem.element.ElementTetP0`
    - :class:`~skfem.element.ElementTetP1`
    - :class:`~skfem.element.ElementTetP2`
    - :class:`~skfem.element.ElementTetRT0`
    - :class:`~skfem.element.ElementTriDG`
    - :class:`~skfem.element.ElementTriP0`
    - :class:`~skfem.element.ElementTriP1`
    - :class:`~skfem.element.ElementTriP2`
    - :class:`~skfem.element.ElementTriRT0`
    - :class:`~skfem.element.ElementVectorH1`
    - :class:`~skfem.element.ElementLineP1`
    - :class:`~skfem.element.ElementLineP2`
    - :class:`~skfem.element.ElementLinePp`
    - :class:`~skfem.element.ElementLineHermite`
    - :class:`~skfem.element.ElementLineMini`

"""

from .discrete_field import DiscreteField
from .element import Element
from .element_h1 import ElementH1
from .element_vector_h1 import ElementVectorH1
from .element_hdiv import ElementHdiv
from .element_hcurl import ElementHcurl
from .element_tri import ElementTriP1, ElementTriP2, ElementTriDG,\
    ElementTriP0, ElementTriRT0, ElementTriMorley,\
    ElementTriArgyris, ElementTriMini
from .element_quad import ElementQuad0, ElementQuad1, ElementQuad2,\
    ElementQuadS2, ElementQuadDG, ElementQuadP,\
    ElementQuadBFS
from .element_tet import ElementTetP0, ElementTetP1, ElementTetP2,\
    ElementTetRT0, ElementTetN0, ElementTetMini
from .element_hex import ElementHex1, ElementHexS2  # noqa
from .element_line import ElementLineP1, ElementLineP2,\
    ElementLinePp, ElementLineHermite, ElementLineMini  # noqa
from .element_composite import ElementComposite  # noqa


__all__ = [
    "DiscreteField",
    "Element",
    "ElementH1",
    "ElementVectorH1",
    "ElementHdiv",
    "ElementHcurl",
    "ElementTriP1",
    "ElementTriP2",
    "ElementTriDG",
    "ElementTriP0",
    "ElementTriRT0",
    "ElementTriMorley",
    "ElementTriArgyris",
    "ElementTriMini",
    "ElementQuad0",
    "ElementQuad1",
    "ElementQuad2",
    "ElementQuadS2",
    "ElementQuadDG",
    "ElementQuadP",
    "ElementQuadBFS",
    "ElementTetP0",
    "ElementTetP1",
    "ElementTetP2",
    "ElementTetRT0",
    "ElementTetN0",
    "ElementTetMini",
    "ElementHex1",
    "ElementHexS2",
    "ElementLineP1",
    "ElementLineP2",
    "ElementLinePp",
    "ElementLineHermite",
    "ElementLineMini",
    "ElementComposite"]
