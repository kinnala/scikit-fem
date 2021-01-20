from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from skfem.mesh import Mesh
from skfem.element import Element
from skfem.mapping import Mapping

from .facet_basis import FacetBasis


class InteriorFacetBasis(FacetBasis):

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 side: int = 0,
                 facets: Optional[ndarray] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None):

        if facets is None:
            facets = np.nonzero(mesh.f2t[1] != -1)[0]

        super(InteriorFacetBasis, self).__init__(mesh,
                                                 elem,
                                                 mapping=mapping,
                                                 intorder=intorder,
                                                 side=side,
                                                 facets=facets,
                                                 quadrature=quadrature)
