from .abstract_basis import AbstractBasis  # noqa
from .cell_basis import CellBasis  # noqa
from .facet_basis import FacetBasis  # noqa
from .interior_facet_basis import InteriorFacetBasis  # noqa

# aliases
Basis = CellBasis
InteriorBasis = CellBasis  # backwards compatibility
ExteriorFacetBasis = FacetBasis  # backwards compatibility
BoundaryFacetBasis = FacetBasis  # backwards compatibility
