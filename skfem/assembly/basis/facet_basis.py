from typing import Tuple, Optional

import numpy as np
from numpy import ndarray

from skfem.element import Element, DiscreteField
from skfem.mapping import Mapping
from skfem.mesh import Mesh
from skfem.quadrature import get_quadrature
from .basis import Basis


class FacetBasis(Basis):
    """Basis functions evaluated at quadrature points on the element boundaries.

    Initialized and used similarly as :class:`~skfem.assembly.InteriorBasis`.

    """
    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 side: Optional[int] = None,
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
            If 0 or 1, basis functions are evaluated on the interior facets.
            The numbers 0 and 1 refer to the different sides of the facets.
            Side 0 corresponds to the indices `mesh.f2t[0]`. If `None`, basis
            is evaluated only on the exterior facets.
        facets
            Optional subset of facet indices.
        quadrature
            Optional tuple of quadrature points and weights.

        """
        super(FacetBasis, self).__init__(mesh, elem, mapping)

        if quadrature is not None:
            self.X, self.W = quadrature
        else:
            self.X, self.W = get_quadrature(
                self.brefdom,
                intorder if intorder is not None else 2 * self.elem.maxdeg
            )

        # facets where the basis is evaluated
        if facets is None:
            if side is None:
                self.find = np.nonzero(self.mesh.f2t[1] == -1)[0]
                self.tind = self.mesh.f2t[0, self.find]
            elif hasattr(self.mapping, 'helper_to_orig') and side in [0, 1]:
                self.mapping.side = side
                self.find = self.mapping.helper_to_orig[side]
                self.tind = self.mesh.f2t[0, self.find]
            elif side in [0, 1]:
                self.find = np.nonzero(self.mesh.f2t[1] != -1)[0]
                self.tind = self.mesh.f2t[side, self.find]
            else:
                raise Exception("Parameter 'side' must be either 0 or 1. "
                                "A facet shares only two elements.")
        else:
            self.find = facets
            self.tind = self.mesh.f2t[0, self.find]

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

    def global_coordinates(self) -> ndarray:
        return DiscreteField(self.mapping.G(self.X, find=self.find))

    def mesh_parameters(self) -> ndarray:
        return DiscreteField((np.abs(self.mapping.detDG(self.X, self.find))
                              ** (1. / (self.mesh.dim() - 1.)))
                             if self.mesh.dim() != 1 else np.array([0.]))

    def trace(self, x: ndarray, elem: Optional[Element] = None):

        from skfem.utils import project

        if elem is None:  # use default element
            from skfem.element import ElementTriP1
            elem = ElementTriP1()

        cls = type(self)
        fbasis = cls(self.mesh,
                     elem,
                     facets=self.find,
                     quadrature=(self.X, self.W))
        I = fbasis.get_dofs(self.find).all()
        if len(I) == 0:  # problem: no facet DOFs
            # solution: take all interior DOFs
            I = fbasis.dofs.interior_dofs[:, self.tind].flatten()
        y = project(x, self, fbasis, I=I)

        # build connectivity for lower dimensional mesh
        ix = self.mesh.facets[:, self.find]
        ixuniq = np.unique(ix)
        b = np.zeros(np.max(ix) + 1, dtype=np.int64)
        b[ixuniq] = np.arange(len(ixuniq), dtype=np.int64)

        def create_mesh(p, t):
            from skfem.mesh import MeshLine
            return MeshLine(p[0], t)

        return create_mesh(self.mesh.p[:, ixuniq], b[ix]), y
