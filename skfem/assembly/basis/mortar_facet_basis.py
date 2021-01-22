from typing import Optional, Tuple

from numpy import ndarray
from skfem.element import Element
from skfem.mapping import MappingMortar
from skfem.mesh import Mesh

from .exterior_facet_basis import ExteriorFacetBasis


class MortarFacetBasis(ExteriorFacetBasis):

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: MappingMortar,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 facets: Optional[ndarray] = None,
                 side: int = 0):
        """Precomputed global basis on the mortar mesh."""

        if side not in (0, 1):
            raise Exception("'side' must be 0 or 1.")

        if facets is None:
            mapping.side = side
            facets = mapping.helper_to_orig[side]

        super(MortarFacetBasis, self).__init__(mesh,
                                               elem,
                                               mapping=mapping,
                                               intorder=intorder,
                                               quadrature=quadrature,
                                               facets=facets,
                                               _side=0)
