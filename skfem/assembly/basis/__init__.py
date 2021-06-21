import warnings

from .abstract_basis import AbstractBasis  # noqa
from .cell_basis import CellBasis  # noqa
from .boundary_facet_basis import BoundaryFacetBasis  # noqa
from .interior_facet_basis import InteriorFacetBasis  # noqa
from .mortar_facet_basis import MortarFacetBasis  # noqa

# aliases
Basis = CellBasis
InteriorBasis = CellBasis  # backwards compatibility
ExteriorFacetBasis = BoundaryFacetBasis  # backwards compatibility


def FacetBasis(*args, side=None, **kwargs):
    """alias of :class:`~skfem.assembly.BoundaryFacetBasis`"""
    if side is None:
        return BoundaryFacetBasis(*args, **kwargs)
    warnings.warn("Initializing FacetBasis using the keyword argument side "
                  "is deprecated. Use InteriorFacetBasis or MortarFacetBasis "
                  "instead.", DeprecationWarning)
    if 'mapping' in kwargs:
        if hasattr(kwargs['mapping'], 'helper_to_orig'):
            return MortarFacetBasis(*args, side=side, **kwargs)
    return InteriorFacetBasis(*args, side=side, **kwargs)
