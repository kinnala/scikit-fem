from typing import Optional, Tuple

from numpy import ndarray
from skfem.element import Element
from skfem.mapping import MappingMortar
from skfem.mesh import Mesh

from .boundary_facet_basis import BoundaryFacetBasis
from ..dofs import Dofs


class MortarFacetBasis(BoundaryFacetBasis):

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: MappingMortar,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 facets: Optional[ndarray] = None,
                 side: int = 0,
                 dofs: Optional[Dofs] = None):
        """Precomputed global basis on the mortar mesh."""

        if side not in (0, 1):
            raise Exception("'side' must be 0 or 1.")

        if facets is None:
            from copy import deepcopy
            mapping = deepcopy(mapping)
            mapping.side = side
            facets = mapping.helper_to_orig[side]

        facets = self._normalize_facets(facets)

        super(MortarFacetBasis, self).__init__(mesh,
                                               elem,
                                               mapping=mapping,
                                               intorder=intorder,
                                               quadrature=quadrature,
                                               facets=facets,
                                               dofs=dofs)
