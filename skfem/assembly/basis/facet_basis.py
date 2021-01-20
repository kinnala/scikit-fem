from typing import Callable, Dict, Optional, Tuple, Type

import numpy as np
from numpy import ndarray

from skfem.element import (DiscreteField, Element, ElementHex0, ElementHex1,
                           ElementHex2, ElementLineP0, ElementLineP1,
                           ElementLineP2, ElementQuad0, ElementQuad1,
                           ElementQuad2, ElementTetP0, ElementTetP1,
                           ElementTetP2, ElementTriP0, ElementTriP1,
                           ElementTriP2)
from skfem.mapping import Mapping
from skfem.mesh import Mesh, MeshHex, MeshQuad, MeshTet, MeshTri

from .basis import Basis
from .interior_basis import InteriorBasis


class FacetBasis(Basis):
    """Basis functions evaluated at quadrature points on the element boundaries.

    Initialized and used similarly as :class:`~skfem.assembly.InteriorBasis`.

    """
    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 side: int = 0,
                 facets: Optional[ndarray] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None):
        """Combine :class:`~skfem.mesh.Mesh` and :class:`~skfem.element.Element`
        into a set of precomputed global basis functions at element facets.

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
        side
            Choose which row of `mesh.f2t` is used to determine the element for
            which the basis function is evaluated. By default, side=0.
        facets
            Optional subset of facet indices.
        quadrature
            Optional tuple of quadrature points and weights.

        """
        super(FacetBasis, self).__init__(mesh,
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
        self.tind = self.mesh.f2t[side, self.find]

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
        from skfem.utils import project

        fbasis = FacetBasis(self.mesh,
                            elem,
                            facets=self.find,
                            quadrature=(self.X, self.W))
        I = fbasis.get_dofs(self.find).all()
        if len(I) == 0:  # special case: no facet DOFs
            if fbasis.dofs.interior_dofs.shape[0] > 1:
                # no one-to-one restriction: requires interpolation
                raise NotImplementedError
            # special case: piecewise constant elem
            I = fbasis.dofs.interior_dofs[:, self.tind].flatten()
        return project(x, self, fbasis, I=I)

    def trace(self,
              x: ndarray,
              projection: Callable[[ndarray], ndarray],
              target_elem: Optional[Element] = None) -> Tuple[InteriorBasis,
                                                              ndarray]:
        """Restrict solution to :math:`d-1` dimensional trace mesh.

        The function ``projection`` defines how the boundary points are
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
            Optional lower-dimensional finite element to project to.  If not
            given, a piecewise constant element is used.

        Returns
        -------
        InteriorBasis
            An object corresponding to the trace mesh.
        ndarray
            A projected solution vector defined on the trace mesh.

        """
        DEFAULT_TARGET = {
            MeshTri: ElementLineP0,
            MeshQuad: ElementLineP0,
            MeshTet: ElementTriP0,
            MeshHex: ElementQuad0,
        }

        meshcls = type(self.mesh)
        if meshcls not in DEFAULT_TARGET:
            raise NotImplementedError("Mesh type not supported.")
        if target_elem is None:
            target_elem = DEFAULT_TARGET[meshcls]()

        ELEMENT_MAP: Dict[Tuple[Type[Element], Type[Mesh]], Type[Element]] = {
            (ElementLineP0, MeshTri): ElementTriP0,
            (ElementLineP1, MeshTri): ElementTriP1,
            (ElementLineP2, MeshTri): ElementTriP2,
            (ElementLineP0, MeshQuad): ElementQuad0,
            (ElementLineP1, MeshQuad): ElementQuad1,
            (ElementLineP2, MeshQuad): ElementQuad2,
            (ElementTriP0, MeshTet): ElementTetP0,
            (ElementTriP1, MeshTet): ElementTetP1,
            (ElementTriP2, MeshTet): ElementTetP2,
            (ElementQuad0, MeshHex): ElementHex0,
            (ElementQuad1, MeshHex): ElementHex1,
            (ElementQuad2, MeshHex): ElementHex2,
        }

        if (type(target_elem), meshcls) not in ELEMENT_MAP:
            raise Exception("The specified 'elem' not supported.")
        elemcls = ELEMENT_MAP[(type(target_elem), meshcls)]
        target_meshcls = type(target_elem).mesh_type

        p, t = self.mesh._reix(self.mesh.facets[:, self.find])

        return (
            InteriorBasis(target_meshcls(projection(p), t), target_elem),
            self._trace_project(x, elemcls())
        )
