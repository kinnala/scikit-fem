import logging
from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray
from skfem.element import (BOUNDARY_ELEMENT_MAP, DiscreteField, Element,
                           ElementHex0, ElementQuad0, ElementTetP0,
                           ElementTriP0)
from skfem.mapping import Mapping
from skfem.mesh import Mesh, MeshHex, MeshLine, MeshQuad, MeshTet, MeshTri

from .abstract_basis import AbstractBasis
from .cell_basis import CellBasis
from ..dofs import Dofs


logger = logging.getLogger(__name__)


class BoundaryFacetBasis(AbstractBasis):
    """For fields defined on the boundary of the domain."""

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 facets: Optional[ndarray] = None,
                 dofs: Optional[Dofs] = None,
                 _tind: Optional[ndarray] = None):
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
        logger.info("Initializing " + typestr)
        super(BoundaryFacetBasis, self).__init__(mesh,
                                                 elem,
                                                 mapping,
                                                 intorder,
                                                 quadrature,
                                                 mesh.brefdom,
                                                 dofs)

        # facets where the basis is evaluated
        if facets is None:
            self.find = np.nonzero(self.mesh.f2t[1] == -1)[0]
        else:
            self.find = self._normalize_facets(facets)

        if len(self.find) == 0:
            logger.warning("Initializing {} with zero facets.".format(typestr))

        if _tind is None:
            self.tind = self.mesh.f2t[0, self.find]
        else:
            self.tind = _tind

        # boundary refdom to global facet
        x = self.mapping.G(self.X, find=self.find)
        # global facet to refdom facet
        Y = self.mapping.invF(x, tind=self.tind)

        # construct normal vectors from side=0 always
        Y0 = self.mapping.invF(x, tind=self.mesh.f2t[0, self.find])
        self.normals = DiscreteField(
            value=self.mapping.normals(Y0,
                                       self.mesh.f2t[0, self.find],
                                       self.find,
                                       self.mesh.t2f)
        )

        self.nelems = len(self.find)

        self.basis = [self.elem.gbasis(self.mapping, Y, j, tind=self.tind)
                      for j in range(self.Nbfun)]

        self.dx = (np.abs(self.mapping.detDG(self.X, find=self.find))
                   * np.tile(self.W, (self.nelems, 1)))
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
        """Trace mesh basis projection."""

        from skfem.utils import projection

        fbasis = BoundaryFacetBasis(self.mesh,
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

    def trace(self,
              x: ndarray,
              projection: Callable[[ndarray], ndarray],
              target_elem: Optional[Element] = None) -> Tuple[CellBasis,
                                                              ndarray]:
        """Restrict solution to :math:`d-1` dimensional trace mesh.

        The parameter ``projection`` defines how the boundary points are
        projected to :math:`d-1` dimensional space.  For example,

        >>> projection = lambda p: p[0]

        will keep only the `x`-coordinate in the trace mesh.

        Parameters
        ----------
        x
            The solution vector.
        projection
            A function defining the projection of the boundary points.  See
            above for an example.
        target_elem
            Optional finite element to project to before restriction.  If not
            given, a piecewise constant element is used.

        Returns
        -------
        CellBasis
            An object corresponding to the trace mesh.
        ndarray
            A projected solution vector defined on the trace mesh.

        """
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

        p, t = self.mesh._reix(self.mesh.facets[:, self.find])

        return (
            CellBasis(target_meshcls(projection(p), t), elemcls()),
            self._trace_project(x, target_elem)
        )

    def with_element(self, elem: Element) -> 'BoundaryFacetBasis':
        """Return a similar basis using a different element."""
        return type(self)(
            self.mesh,
            elem,
            mapping=self.mapping,
            quadrature=self.quadrature,
            facets=self.find,
        )

    def project(self, interp, facets=None):
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

        M, f = self._projection(interp)

        if facets is not None:
            return solve(*condense(M, f, I=self.get_dofs(facets=facets)))
        return solve(*condense(M, f, I=self.get_dofs(facets=self.find)))
