from typing import Optional, Tuple, Any

from numpy import ndarray
from skfem.element import Element
from skfem.mapping import MappingMortar
from skfem.mesh import Mesh

from .facet_basis import FacetBasis
from ..dofs import Dofs


class MortarFacetBasis(FacetBasis):

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: MappingMortar,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 facets: Optional[Any] = None,
                 dofs: Optional[Dofs] = None,
                 side: int = 0):
        """Precomputed global basis on the mortar mesh."""

        if side not in (0, 1):
            raise Exception("'side' must be 0 or 1.")

        if facets is None:
            from copy import deepcopy
            mapping = deepcopy(mapping)
            mapping.side = side
            facets = mapping.helper_to_orig[side]

        facets = mesh.normalize_facets(facets)

        super(MortarFacetBasis, self).__init__(
            mesh,
            elem,
            mapping=mapping,
            intorder=intorder,
            quadrature=quadrature,
            facets=facets,
            dofs=dofs,
        )

    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        return {
            **super(MortarFacetBasis, self).default_parameters(),
            # TODO following is not valid in 3D
            'h1': self.mapping.maps[0].detDG(self.X,
                                             self.mapping.helper_to_orig[0]),
            'h2': self.mapping.maps[1].detDG(self.X,
                                             self.mapping.helper_to_orig[1]),
        }
