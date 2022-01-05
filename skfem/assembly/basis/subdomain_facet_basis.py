import logging
from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from skfem.element import Element, ElementTriP0
from skfem.mapping import Mapping
from skfem.mesh import Mesh

from .. import asm
from .cell_basis import CellBasis
from .facet_basis import FacetBasis
from .interior_facet_basis import InteriorFacetBasis
from ..dofs import Dofs
from ..form import Functional

logger = logging.getLogger(__name__)


class SubdomainFacetBasis(FacetBasis):
    def __init__(
        self,
        mesh: Mesh,
        elem: Element,
        elements,
        side: int = 0,
        mapping: Optional[Mapping] = None,
        intorder: Optional[int] = None,
        quadrature: Optional[Tuple[ndarray, ndarray]] = None,
        facets: Optional[ndarray] = None,
        dofs: Optional[Dofs] = None,
    ):
        def get_boundary(facet_list, flip_list):
            def _form(w):
                for i, indicator in zip(
                    np.nonzero(mesh.f2t[1] != -1)[0], w.ind_p0_s0 - w.ind_p0_s1
                ):
                    if indicator[0] != 0 and i not in facet_list:
                        facet_list.append(i)
                        flip_list.append(indicator)
                return w.x[0]  # must return *something* of the right shape

            return Functional(_form)

        basis_p0 = CellBasis(mesh, ElementTriP0())
        facet_basis_p0_s0 = InteriorFacetBasis(mesh, ElementTriP0(), side=0)
        facet_basis_p0_s1 = InteriorFacetBasis(mesh, ElementTriP0(), side=1)

        ind_p0 = basis_p0.zeros()
        ind_p0[basis_p0.get_dofs(elements=elements)] = 1

        facet_list = list()
        flip_list = list()
        asm(
            get_boundary(facet_list, flip_list),
            [facet_basis_p0_s0, facet_basis_p0_s1],
            ind_p0_s0=facet_basis_p0_s0.interpolate(ind_p0),
            ind_p0_s1=facet_basis_p0_s1.interpolate(ind_p0),
        )

        self.mesh = mesh  # required by _normalize_elements
        elements = self._normalize_elements(elements)

        facets = np.array(facet_list)
        tind = mesh.f2t[side, facets]

        super().__init__(
            mesh,
            elem,
            mapping=mapping,
            intorder=intorder,
            quadrature=quadrature,
            facets=facets,
            _tind=tind,
            dofs=dofs,
        )

        self.normals *= np.array(flip_list)
        if side == 1:
            self.normals *= -1