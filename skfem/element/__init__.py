"""The module :mod:`skfem.element` defines finite elements in a very generic
sense.

In order to use an element, you simply initialize the respective object and
pass it to the constructor of :class:`~skfem.assembly.InteriorBasis` or
:class:`~skfem.assembly.FacetBasis`.  See below for a list of supported
elements.

Choosing a finite element
-------------------------

Here are some general instructions for choosing an :class:`Element` class.
Firstly, the naming of the element classes reflects their compatibility with
the mesh types:

>>> from skfem.element import ElementTriP1
>>> ElementTriP1.mesh_type
<class 'skfem.mesh.mesh2d.mesh_tri.MeshTri'>

Secondly, the chosen finite element should be compatible with the approximated
partial differential equation.  Stability of the finite element approximations
is a broad topic and will not be covered here.  However, here are some general
rules:

* use subclasses of :class:`ElementH1` for standard second-order problems
* you can discretize vectorial problems by manually building the block matrices
  (e.g., using ``scipy.sparse.bmat``) with scalar elements, or by using
  :class:`ElementVectorH1` and :class:`ElementComposite` that abstract out the
  creation of the block matrices
* pay special attention to constrained problems, e.g., the Stokes system which
  may require the use of elements such as :class:`ElementTriMini`
* use subclasses of :class:`ElementHdiv` or :class:`ElementHcurl`, e.g.,
  :class:`ElementTriRT0` or :class:`ElementTetN0`, for mixed problems with less
  regular solutions
* use subclasses of :class:`ElementGlobal`, e.g., :class:`ElementTriMorley` or
  :class:`ElementTriArgyris`, for fourth-order problems or if there are special
  postprocessing requirements, e.g., the need for high-order derivatives.

Thirdly, the different finite element spaces use different degrees-of-freedom.
It is up to the user to decide whether the given boundary condition can or
should be enforced strongly or weakly.  See :ref:`finddofs` for more
information.

List of elements
----------------

"""

from .discrete_field import DiscreteField
from .element import Element
from .element_h1 import ElementH1
from .element_vector_h1 import ElementVectorH1
from .element_hdiv import ElementHdiv
from .element_hcurl import ElementHcurl
from .element_tri import (ElementTriP1, ElementTriP2, ElementTriDG,
                          ElementTriP0, ElementTriRT0, ElementTriMorley,
                          ElementTriArgyris, ElementTriMini, ElementTriCR,
                          ElementTriHermite)
from .element_quad import (ElementQuad0, ElementQuad1, ElementQuad2,
                           ElementQuadS2, ElementQuadDG, ElementQuadP,
                           ElementQuadBFS)
from .element_tet import (ElementTetP0, ElementTetP1, ElementTetP2,
                          ElementTetRT0, ElementTetN0, ElementTetMini,
                          ElementTetCR)
from .element_hex import (ElementHex0, ElementHex1, ElementHex2,
                          ElementHexS2)  # noqa
from .element_line import (ElementLineP0, ElementLineP1, ElementLineP2,
                           ElementLinePp, ElementLineHermite,
                           ElementLineMini)  # noqa
from .element_composite import ElementComposite  # noqa


__all__ = [
    "DiscreteField",
    "Element",
    "ElementH1",
    "ElementVectorH1",
    "ElementHdiv",
    "ElementHcurl",
    "ElementComposite",
    "ElementTriP1",
    "ElementTriP2",
    "ElementTriDG",
    "ElementTriP0",
    "ElementTriCR",
    "ElementTriRT0",
    "ElementTriMorley",
    "ElementTriArgyris",
    "ElementTriMini",
    "ElementTriHermite",
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
    "ElementTetCR",
    "ElementHex0",
    "ElementHex1",
    "ElementHex2",
    "ElementHexS2",
    "ElementLineP0",
    "ElementLineP1",
    "ElementLineP2",
    "ElementLinePp",
    "ElementLineHermite",
    "ElementLineMini",
]
