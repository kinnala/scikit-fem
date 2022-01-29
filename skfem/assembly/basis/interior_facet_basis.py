from typing import Optional, Tuple, Any

import numpy as np
from numpy import ndarray
from skfem.element import Element
from skfem.mapping import Mapping
from skfem.mesh import Mesh

from .facet_basis import FacetBasis
from ..dofs import Dofs


class InteriorFacetBasis(FacetBasis):
    """For evaluating integrals over interior facets.

    Useful for, e.g., a posteriori error estimators or implementing interior
    penalty/discontinuous Galerkin methods.

    """
    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 facets: Optional[Any] = None,
                 dofs: Optional[Dofs] = None,
                 side: int = 0):
        """Precomputed global basis on interior facets."""

        if facets is None:
            facets = np.nonzero(mesh.f2t[1] != -1)[0]

        facets = mesh.normalize_facets(facets)

        super(InteriorFacetBasis, self).__init__(
            mesh,
            elem,
            mapping=mapping,
            intorder=intorder,
            quadrature=quadrature,
            facets=facets,
            dofs=dofs,
            side=side,
        )
