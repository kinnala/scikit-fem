from typing import Optional

import numpy as np
from numpy import ndarray

from skfem.element import DiscreteField
from skfem.quadrature import get_quadrature
from .basis import Basis


class FacetBasis(Basis):
    """Global basis functions evaluated at integration points on the element
    boundaries.

    Examples
    --------
    :class:`~skfem.assembly.FacetBasis` object is a combination of
    :class:`~skfem.mesh.Mesh`, :class:`~skfem.element.Element`, and
    :class:`~skfem.mapping.Mapping`:

    >>> from skfem import *
    >>> from skfem.models.poisson import mass
    >>> m = MeshTri.init_symmetric()
    >>> e = ElementTriP1()
    >>> fb = FacetBasis(m, e, MappingAffine(m))

    The object is used in the assembly of bilinear and
    linear forms where the integral is over the boundary
    of the domain (or elements).

    >>> B = asm(mass, fb)
    >>> B.shape
    (5, 5)

    """
    def __init__(self,
                 mesh,
                 elem,
                 mapping = None,
                 intorder: Optional[int] = None,
                 side: Optional[int] = None,
                 facets: Optional[ndarray] = None):
        """Combine :class:`~skfem.mesh.Mesh` and :class:`~skfem.element.Element`
        into a set of precomputed global basis functions at element facets.

        Parameters
        ----------
        mesh
            An object of type :class:`~skfem.mesh.Mesh`.
        elem
            An object of type :class:`~skfem.element.Element`.
        mapping
            An object of type :class:`~skfem.mapping.Mapping`.
        intorder
            Optional integration order, i.e. the degree of polynomials that are
            integrated exactly by the used quadrature.
        side
            If 0 or 1, the basis functions are evaluated on the interior facets.
            The numbers 0 and 1 refer to the different sides of the facets.
            Side 0 corresponds to the indices mesh.f2t[0, :].
        facets
            Optional subset of facet indices.

        """
        super(FacetBasis, self).__init__(mesh, elem, mapping, intorder)

        self.X, self.W = get_quadrature(self.brefdom, self.intorder)

        # facets where the basis is evaluated
        if facets is None:
            if side is None:
                self.find = np.nonzero(self.mesh.f2t[1] == -1)[0]
                self.tind = self.mesh.f2t[0, self.find]
            elif hasattr(self.mapping, 'helper_to_orig') and side in [0, 1]:
                self.mapping.side = side # side effect
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
            value = self.mapping.normals(Y0,
                                         self.mesh.f2t[0, self.find],
                                         self.find,
                                         self.mesh.t2f)
        )

        self.nelems = len(self.find)

        self.basis = [self.elem.gbasis(self.mapping, Y, j, self.tind)
                      for j in range(self.Nbfun)]

        self.dx = (np.abs(self.mapping.detDG(self.X, find=self.find))
                   * np.tile(self.W, (self.nelems, 1)))

        self.element_dofs = self.element_dofs[:, self.tind]

    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        return {'x': self.global_coordinates(),
                'h': self.mesh_parameters(),
                'n': self.normals}

    def global_coordinates(self) -> ndarray:
        return DiscreteField(self.mapping.G(self.X, find=self.find))

    def mesh_parameters(self) -> ndarray:
        return DiscreteField((np.abs(self.mapping.detDG(self.X, self.find))
                              ** (1. / (self.mesh.dim() - 1.)))\
                             if self.mesh.dim() != 1 else np.array([0.]))
