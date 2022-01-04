from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from skfem.element import Element
from skfem.mapping import Mapping
from skfem.mesh import Mesh

from .facet_basis import FacetBasis
from ..dofs import Dofs


class BoundaryFacetBasis(FacetBasis):
    """For fields defined on the boundary of the domain."""

    pass
