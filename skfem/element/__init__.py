"""This module defines finite elements (in a very generic sense).  The naming of
the element classes reflects their compatibility with the mesh types.

>>> from skfem.element import ElementTriP1
>>> ElementTriP1.mesh_type
skfem.mesh.mesh2d.mesh_tri.MeshTri

In order to use an element, you simply initialize the respective object and pass
it to the constructor of :class:`~skfem.assembly.InteriorBasis` or
:class:`~skfem.assembly.FacetBasis`.

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
    ElementTetRT0, ElementTetN0
from .element_hex import ElementHex1
from .element_line import ElementLineP1, ElementLineP2, ElementLinePp,\
    ElementLineHermite
from .element_composite import ElementComposite
