import numpy as np
from numpy import ndarray

from ...quadrature import get_quadrature
from .facet_basis import FacetBasis
from .interior_basis import InteriorBasis
from ...element import DiscreteField


class MortarBasis(FacetBasis):
    """Global basis functions at integration points on the mortar boundary."""

    def __init__(self,
                 mesh,
                 elem,
                 mapping,
                 intorder):
        """Initialize a basis for assembling mortar matrices.

        Parameters
        ----------
        mesh
            An object of type :class:`~skfem.mesh.Mesh`.
        elem
            An object of type :class:`~skfem.element.Element`.
        mapping
            Mapping to the relevant facets of the mesh, see
            :class:`~skfem.mapping.MortarPair`.
        intorder
            Integration order, i.e. the degree of polynomials that are
            integrated exactly by the used quadrature. Please use equivalent
            integration orders on both sides of the mortar boundary.

        """
        super(FacetBasis, self).__init__(mesh, elem, mapping, intorder)

        self.X, self.W = get_quadrature(self.brefdom, self.intorder)

        # global facets where basis is evaluated
        self.find = self.mapping.find
        self.tind = self.mesh.f2t[0, self.find]

        # boundary refdom to global facet
        x = self.mapping.G(self.X)

        # global facet to refdom facet
        Y = self.mapping.invF(x, tind=self.tind)

        # normals are defined in the mortar mapping
        self.normals = np.repeat(self.mapping.normals[:, :, None],
                                 len(self.W),
                                 axis=2)
        self.normals = DiscreteField(self.normals)

        self.basis = [self.elem.gbasis(self.mapping, Y, j, self.tind)
                      for j in range(self.Nbfun)]

        self.nelems = len(self.find)
        self.dx = np.abs(self.mapping.detDG(self.X, find=self.find)) *\
            np.tile(self.W, (self.nelems, 1))

        self.element_dofs = self.element_dofs[:, self.tind]
