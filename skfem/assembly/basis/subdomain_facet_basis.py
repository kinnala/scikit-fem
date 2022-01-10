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
                 elements: ndarray,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 side: int = 0,
                 dofs: Optional[Dofs] = None):
        # Used by .with_element()
        self._elements = elements
        self._side = side

        elements = mesh.normalize_elements(elements)
        # Get indices of all facets in the mesh
        all_facets, all_counts = np.unique(mesh.t2f,
                                           return_counts=True)
        # Find the exterior ones
        exterior_facets = all_facets[all_counts == 1]
        # Get indices of all facets in the subdomain
        subdomain_facets, sub_counts = np.unique(mesh.t2f[:, elements],
                                                 return_counts=True)
        # Don't want facets on the exterior boundary, even if they are on the
        # subdomain boundary
        not_exterior = np.logical_not(
                np.isin(subdomain_facets, exterior_facets))
        # Find facets that are on boundary of subdomain (but not boundary of
        # exterior)
        facets = subdomain_facets[np.logical_and(sub_counts == 1,
                                                 not_exterior)]
        # Find the indices of the triangles for these facets (note some of
        # these triangles won't be in the subdomain.)
        tind = mesh.f2t[:, facets].flatten('F')
        # Boolean indicator array for which triangles are in the subdomain
        ix = np.in1d(tind, elements)
        # Get triangles to use for this basis based on `side` parameter
        _tind = tind[~ix] if side == 1 else tind[ix]
        super().__init__(
            mesh,
            elem,
            mapping=mapping,
            intorder=intorder,
            quadrature=quadrature,
            facets=facets,
            dofs=dofs,
            _tind=_tind,
            _tind_normals=_tind,
        )

    def with_element(self, elem: Element) -> 'SubdomainFacetBasis':
        """Return a similar basis using a different element."""
        return type(self)(
            self.mesh,
            elem,
            elements=self._elements,
            side=self._side,
            mapping=self.mapping,
            quadrature=self.quadrature,
        )
