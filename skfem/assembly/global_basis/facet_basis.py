from typing import Dict, Optional

import numpy as np
from numpy import ndarray

from skfem.quadrature import get_quadrature

from .global_basis import GlobalBasis


class FacetBasis(GlobalBasis):
    """Global basis functions evaluated at integration points on the element
    boundaries.

    Attributes
    ----------
    phi : ndarray
        Global basis functions at global quadrature points.
    dphi : ndarray
        Global basis function derivatives at global quadrature points.
    X : ndarray
        Local quadrature points (Ndim x Nqp).
    W : ndarray
        Local quadrature weights (Nqp).
    nf : int
        Number of facets.
    dx : ndarray
        Used in computing global integrals elementwise (Nelems x Nqp).
        For example, np.sum(u**2*dx, axis=1) where u is also
        a numpy array of size Nelems x Nqp.
    find : ndarray
        A list of facet indices.
    tind : ndarray
        A list of triangle indices.
    normals : ndarray
    mapping : skfem.mapping.Mapping
    elem : skfem.element.Element
    Nbfun : int
        The number of basis functions.
    intorder : int
        The integration order.
    dim : int
        The problem dimension.
    nt : int
        Number of triangles.
    mesh : skfem.mesh.Mesh
    refdom : string
    brefdom : string

    Examples
    --------

    FacetBasis object is a combination of Mesh, Element,
    and Mapping:

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
                 mapping=None,
                 intorder: Optional[int] = None,
                 side: Optional[int] = None,
                 facets = None):
        super(FacetBasis, self).__init__(mesh, elem, mapping, intorder)

        self.X, self.W = get_quadrature(self.brefdom, self.intorder)

        # facets where the basis is evaluated
        if facets is None:
            if side is None:
                self.find = np.nonzero(self.mesh.f2t[1, :] == -1)[0]
                self.tind = self.mesh.f2t[0, self.find]
            elif side == 0 or side == 1:
                self.find = np.nonzero(self.mesh.f2t[1, :] != -1)[0]
                self.tind = self.mesh.f2t[side, self.find]
            else:
                raise Exception("Parameter side must be 0 or 1. Facet shares only two elements.")
        else:
            self.find = facets
            self.tind = self.mesh.f2t[0, self.find]

        # boundary refdom to global facet
        x = self.mapping.G(self.X, find=self.find)
        # global facet to refdom facet
        Y = self.mapping.invF(x, tind=self.tind)

        if hasattr(mesh, 'normals'):
            self.normals = np.repeat(mesh.normals[:, :, None], len(self.W), axis=2)
        else:
            # construct normal vectors from side=0 always
            Y0 = self.mapping.invF(x, tind=self.mesh.f2t[0, self.find]) # TODO check why without this works also (Y0 = Y)
            self.normals = self.mapping.normals(Y0, self.mesh.f2t[0, self.find], self.find, self.mesh.t2f)

        self.nelems = len(self.find)

        self.basis = list(zip(*[self.elem.gbasis(self.mapping, Y, j, self.tind) for j in range(self.Nbfun)]))

        self.dx = np.abs(self.mapping.detDG(self.X, find=self.find)) * np.tile(self.W, (self.nelems, 1))

        self.element_dofs = self.element_dofs[:, self.tind] # TODO this is required for asm(). Check for other options.

    def default_parameters(self) -> Dict[str, ndarray]:
        return {'x':self.global_coordinates(),
                'h':self.mesh_parameters(),
                'n':self.normals}
    
    def global_coordinates(self) -> ndarray:
        return self.mapping.G(self.X, find=self.find)

    def mesh_parameters(self) -> ndarray:
        if self.mesh.dim() == 1:
            return np.array([0.0])
        else:
            return np.abs(self.mapping.detDG(self.X, self.find)) ** (1.0 / (self.mesh.dim() - 1))
