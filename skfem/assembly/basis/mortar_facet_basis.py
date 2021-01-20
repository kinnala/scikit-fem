from typing import Optional, Tuple

from numpy import ndarray
from skfem.element import Element
from skfem.mapping import MappingMortar
from skfem.mesh import Mesh

from .facet_basis import FacetBasis


class MortarFacetBasis(FacetBasis):

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: MappingMortar,
                 intorder: Optional[int] = None,
                 side: int = 0,
                 facets: Optional[ndarray] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None):

        if facets is None:
            mapping.side = side
            facets = mapping.helper_to_orig[side]

        super(MortarFacetBasis, self).__init__(mesh,
                                               elem,
                                               mapping=mapping,
                                               intorder=intorder,
                                               side=0,
                                               facets=facets,
                                               quadrature=quadrature)
