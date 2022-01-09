from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from skfem.element import Element
from skfem.mapping import Mapping
from skfem.mesh import Mesh

from .boundary_facet_basis import BoundaryFacetBasis
from ..dofs import Dofs


class SubdomainFacetBasis(BoundaryFacetBasis):

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 elements: Optional[ndarray] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 side: int = 0,
                 dofs: Optional[Dofs] = None):

        assert elements is not None
        self.mesh = mesh  # required by _normalize_elements
        elements = self._normalize_elements(elements)
        assert isinstance(elements, ndarray)
        all_facets, counts = np.unique(mesh.t2f[:, elements],
                                       return_counts=True)
        facets = all_facets[counts == 1]
        tind = mesh.f2t[:, facets].flatten('F')
        ix = np.in1d(tind, elements)
        _tind = tind[~ix] if side == 1 else tind[ix]
        super().__init__(
            mesh,
            elem,
            mapping=mapping,
            intorder=intorder,
            quadrature=quadrature,
            facets=facets,
            _tind=_tind,
            _tind_normals=_tind,
            dofs=dofs,
        )
