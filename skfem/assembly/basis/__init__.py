from .abstract_basis import AbstractBasis  # noqa
from .cell_basis import CellBasis  # noqa
from .facet_basis import FacetBasis  # noqa
from .interior_facet_basis import InteriorFacetBasis  # noqa
from skfem.generic_utils import Removed

# aliases
Basis = CellBasis
InteriorBasis = Removed("skfem.InteriorBasis",
                        version="13.0.0",
                        era="pre-4.0",
                        message="Use Basis instead of InteriorBasis.")
ExteriorFacetBasis = Removed("skfem.ExteriorFacetBasis",
                             version="13.0.0",
                             era="pre-4.0",
                             message=("Use FacetBasis instead of "
                                      "ExteriorFacetBasis."))
BoundaryFacetBasis = Removed("skfem.BoundaryFacetBasis",
                             version="13.0.0",
                             era="pre-4.0",
                             message=("Use FacetBasis instead of "
                                      "BoundaryFacetBasis."))
