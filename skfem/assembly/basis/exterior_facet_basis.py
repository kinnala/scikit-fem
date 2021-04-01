from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray
from skfem.element import (BOUNDARY_ELEMENT_MAP, DiscreteField, Element,
                           ElementHex0, ElementQuad0, ElementTetP0,
                           ElementTriP0)
from skfem.mapping import Mapping
from skfem.mesh import Mesh, MeshHex, MeshLine, MeshQuad, MeshTet, MeshTri

from .basis import Basis
from .interior_basis import InteriorBasis


class ExteriorFacetBasis(Basis):
    """Global basis functions at quadrature points on the boundary.

    Aliased as ``FacetBasis``.

    """
    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 facets: Optional[ndarray] = None,
                 _side: int = 0):
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

        """
        super(ExteriorFacetBasis, self).__init__(mesh,
                                                 elem,
                                                 mapping,
                                                 intorder,
                                                 quadrature,
                                                 mesh.brefdom)

        # facets where the basis is evaluated
        if facets is None:
            self.find = np.nonzero(self.mesh.f2t[1] == -1)[0]
        else:
            self.find = facets
        self.tind = self.mesh.f2t[_side, self.find]

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

    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        return {'x': self.global_coordinates(),
                'h': self.mesh_parameters(),
                'n': self.normals}

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

        fbasis = ExteriorFacetBasis(self.mesh,
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
              target_elem: Optional[Element] = None) -> Tuple[InteriorBasis,
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
        InteriorBasis
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

        p, t = self.mesh._reix(self.mesh.facets[:, self.find])

        return (
            InteriorBasis(target_meshcls(projection(p), t), elemcls()),
            self._trace_project(x, target_elem)
        )

    def with_element(self, elem: Element) -> 'ExteriorFacetBasis':
        """Return a similar basis using a different element."""
        return type(self)(
            self.mesh,
            elem,
            mapping=self.mapping,
            quadrature=self.quadrature,
            facets=self.find,
        )
