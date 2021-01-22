from .basis import Basis  # noqa
from .interior_basis import InteriorBasis  # noqa
from .exterior_facet_basis import ExteriorFacetBasis  # noqa
from .interior_facet_basis import InteriorFacetBasis  # noqa
from .mortar_facet_basis import MortarFacetBasis  # noqa


def FacetBasis(*args, side=None, **kwargs):
    """For backwards compatibility."""
    if side is None:
        return ExteriorFacetBasis(*args, **kwargs)
    if 'mapping' in kwargs:
        if hasattr(kwargs['mapping'], 'helper_to_orig'):
            return MortarFacetBasis(*args, side=side, **kwargs)
    return InteriorFacetBasis(*args, side=side, **kwargs)
