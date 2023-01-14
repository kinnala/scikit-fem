import logging
from typing import Callable, Optional, Tuple, Any

import numpy as np
from numpy import ndarray
from skfem.element import (BOUNDARY_ELEMENT_MAP, DiscreteField, Element,
                           ElementHex0, ElementQuad0, ElementTetP0,
                           ElementTriP0)
from skfem.mapping import Mapping
from skfem.mesh import Mesh, MeshHex, MeshLine, MeshQuad, MeshTet, MeshTri
from skfem.generic_utils import OrientedBoundary, deprecated

from .abstract_basis import AbstractBasis
from .cell_basis import CellBasis
from ..dofs import Dofs


logger = logging.getLogger(__name__)


class FacetBasis(AbstractBasis):
    """For integrating over facets of the mesh.  Usually over the boundary."""

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 facets: Optional[Any] = None,
                 dofs: Optional[Dofs] = None,
                 side: int = 0):
        """Precomputed global basis on boundary facets.

        Parameters
        ----------
        mesh
            An object of type :class:`~skfem.mesh.Mesh`.
        elem
            An object of type :class:`~skfem.element.Element`.
        mapping
            An object of type :class:`skfem.mapping.Mapping`. If `None`, uses
            `mesh.mapping`.
        intorder
            Optional integration order, i.e. the degree of polynomials that are
            integrated exactly by the used quadrature. Not used if `quadrature`
            is specified.
        quadrature
            Optional tuple of quadrature points and weights.
        facets
            Optional subset of facet indices.
        dofs
            Optional :class:`~skfem.assembly.Dofs` object.

        """
        typestr = ("{}({}, {})".format(type(self).__name__,
                                       type(mesh).__name__,
                                       type(elem).__name__))
        logger.info("Initializing {}".format(typestr))
        super(FacetBasis, self).__init__(
            mesh,
            elem,
            mapping,
            intorder,
            quadrature,
            mesh.brefdom,
            dofs,
        )

        # by default use boundary facets
        if facets is None:
            self.find = np.nonzero(self.mesh.f2t[1] == -1)[0]
        else:
            self.find = mesh.normalize_facets(facets)

        # fix the orientation
        if isinstance(self.find, OrientedBoundary):
            self.tind = self.mesh.f2t[(-1) ** side * self.find.ori - side,
                                      self.find]
            self.tind_normals = self.mesh.f2t[self.find.ori, self.find]
        else:
            self.tind = self.mesh.f2t[side, self.find]
            self.tind_normals = self.mesh.f2t[0, self.find]

        if len(self.find) == 0:
            logger.warning("Initializing {} with no facets.".format(typestr))

        # boundary refdom to global facet
        x = self.mapping.G(self.X, find=self.find)
        # global facet to refdom facet
        Y = self.mapping.invF(x, tind=self.tind)

        # calculate normals
        Y0 = self.mapping.invF(x, tind=self.tind_normals)
        assert self.tind_normals is not None  # satisfy mypy
        self.normals = DiscreteField(
            value=self.mapping.normals(Y0,
                                       self.tind_normals,
                                       self.find,
                                       self.mesh.t2f)
        )

        self.nelems = len(self.find)

        self.basis = [self.elem.gbasis(self.mapping, Y, j, tind=self.tind)
                      for j in range(self.Nbfun)]

        self.dx = (np.abs(self.mapping.detDG(self.X, find=self.find))
                   * np.broadcast_to(self.W, (self.nelems, self.W.shape[-1])))
        logger.info("Initializing finished.")

    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        return {
            'x': self.global_coordinates(),
            'h': self.mesh_parameters(),
            'n': self.normals,
        }

    def global_coordinates(self) -> DiscreteField:
        return DiscreteField(self.mapping.G(self.X, find=self.find))

    def mesh_parameters(self) -> DiscreteField:
        return DiscreteField((np.abs(self.mapping.detDG(self.X, self.find))
                              ** (1. / (self.mesh.dim() - 1.)))
                             if self.mesh.dim() != 1 else np.array([0.]))

    def with_element(self, elem: Element) -> 'FacetBasis':
        """Return a similar basis using a different element."""
        return type(self)(
            self.mesh,
            elem,
            mapping=self.mapping,
            quadrature=self.quadrature,
            facets=self.find,
        )

    def project(self, interp, facets=None, dtype=None):
        """Perform :math:`L^2` projection onto the basis on the boundary.

        The values of the interior DOFs remain zero.

        Parameters
        ----------
        interp
            An object of type :class:`~skfem.element.DiscreteField` which is a
            function (to be projected) evaluated at global quadrature points at
            the boundary of the domain.  If a function is given, then
            :class:`~skfem.element.DiscreteField` is created by passing
            an array of global quadrature point locations to the function.
        facets
            Optionally perform the projection on a subset of facets.  The
            values of the remaining DOFs are zero.

        """
        from skfem.utils import solve, condense

        M, f = self._projection(interp, dtype=dtype)

        if facets is not None:
            return solve(*condense(M, f, I=self.get_dofs(facets=facets)))
        return solve(*condense(M, f, I=self.get_dofs(facets=self.find)))
