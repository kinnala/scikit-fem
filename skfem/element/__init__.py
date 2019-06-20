"""Element classes define and evaluate the finite element basis
functions.

:class:`~skfem.element.Element` objects are supplemented to the
constructors of :class:`~skfem.assembly.InteriorBasis` and
:class:`~skfem.assembly.FacetBasis` by the user.

>>> from skfem import *
>>> m = MeshTri()
>>> e = ElementTriP2()
>>> basis = InteriorBasis(m, e)

"""

from .element import Element
from .element_h1 import ElementH1
from .element_vector_h1 import ElementVectorH1
from .element_hdiv import ElementHdiv
from .element_hcurl import ElementHcurl
from .element_h2 import ElementH2
from .element_tri import ElementTriP1, ElementTriP2, ElementTriDG,\
    ElementTriP0, ElementTriRT0, ElementTriMorley,\
    ElementTriArgyris
from .element_quad import ElementQuad1, ElementQuad2
from .element_tet import ElementTetP0, ElementTetP1, ElementTetP2,\
    ElementTetRT0, ElementTetN0
from .element_hex import ElementHex1
from .element_line import ElementLineP1

