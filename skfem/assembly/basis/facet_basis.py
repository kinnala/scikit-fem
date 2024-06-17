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
                 side: int = 0,
                 disable_doflocs: bool = False):
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
        disable_doflocs
            If `True`, the computation of global DOF locations is
            disabled.  This may save memory on large meshes if DOF
            locations are not required.

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
            disable_doflocs,
        )

        # by default use boundary facets
        if facets is None:
            self.find = np.nonzero(self.mesh.f2t[1] == -1)[0].astype(np.int32)
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

    def _trace_project(self,
                       x: ndarray,
                       elem: Element) -> ndarray:
        from skfem.utils import projection

        fbasis = FacetBasis(self.mesh,
                            elem,
                            facets=self.find,
                            quadrature=(self.X, self.W))
        I = fbasis.get_dofs(self.find).all()
        if len(I) == 0:  # special case: no facet DOFs
            if fbasis.dofs.interior_dofs is not None:
                if fbasis.dofs.interior_dofs.shape[0] > 1:
                    # no one-to-one restriction: requires interpolation
                    raise NotImplementedError
                # special case: piecewise constant elem
                I = fbasis.dofs.interior_dofs[:, self.tind].flatten()
            else:
                raise ValueError
        return projection(x, fbasis, self, I=I)

    @deprecated("Basis.interpolator + Basis.project")
    def trace(self,
              x: ndarray,
              projection: Callable[[ndarray], ndarray],
              target_elem: Optional[Element] = None) -> Tuple[CellBasis,
                                                              ndarray]:

        DEFAULT_TARGET = {
            MeshTri: ElementTriP0,
            MeshQuad: ElementQuad0,
            MeshTet: ElementTetP0,
            MeshHex: ElementHex0,
        }

        meshcls = type(self.mesh)
        if meshcls not in DEFAULT_TARGET:
            raise NotImplementedError("Mesh type not supported.")
        if target_elem is None:
            target_elem = DEFAULT_TARGET[meshcls]()

        if type(target_elem) not in BOUNDARY_ELEMENT_MAP:
            raise Exception("The specified element not supported.")
        elemcls = BOUNDARY_ELEMENT_MAP[type(target_elem)]
        target_meshcls = {
            MeshTri: MeshLine,
            MeshQuad: MeshLine,
            MeshTet: MeshTri,
            MeshHex: MeshQuad,
        }[meshcls]

        assert callable(target_meshcls)  # to satisfy mypy

        p, t, _ = self.mesh._reix(self.mesh.facets[:, self.find])

        return (
            CellBasis(target_meshcls(projection(p), t), elemcls()),
            self._trace_project(x, target_elem)
        )

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
        dtype
            Set to `np.complex64` or similar to use complex numbers.

        """
        from skfem.utils import solve, condense

        M, f = self._projection(interp, dtype=dtype)

        if facets is not None:
            return solve(*condense(M, f, I=self.get_dofs(facets=facets)))
        return solve(*condense(M, f, I=self.get_dofs(facets=self.find)))
