from .abstract_basis import AbstractBasis  # noqa
from .cell_basis import CellBasis  # noqa
from .boundary_facet_basis import BoundaryFacetBasis  # noqa
from .interior_facet_basis import InteriorFacetBasis  # noqa
from .mortar_facet_basis import MortarFacetBasis  # noqa

# aliases
Basis = CellBasis
InteriorBasis = CellBasis  # backwards compatibility
ExteriorFacetBasis = BoundaryFacetBasis  # backwards compatibility
FacetBasis = BoundaryFacetBasis
