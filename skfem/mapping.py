"""Mappings define relationships between reference and global elements.

:class:`~skfem.mesh.Mesh` provides default mappings for each mesh type,
so normally the user is not required to access these classes.

"""

import numpy as np

from typing import Optional

from numpy import ndarray
from skfem.mesh import Mesh
from skfem.element import Element

class Mapping():
    def F(self,
          X: ndarray,
          tind: Optional[ndarray] = None) -> ndarray:
        """Perform mapping from the reference element
        to global elements.

        Parameters
        ----------
        X
            Local points on the reference element (Ndim x Nqp).
        tind
            A set of element indices to map to

        Returns
        -------
        ndarray
            Global points (Ndim x Nelems x Nqp)

        """
        raise NotImplementedError("!")

    def invF(self,
             x: ndarray,
             tind: Optional[ndarray] = None) -> ndarray:
        """Perform an inverse mapping from global elements to
        reference element.

        Parameters
        ----------
        x
            The global points (Ndim x Nelems x Nqp).
        tind
            A set of element indices to map from

        Returns
        -------
        ndarray
            The corresponding local points (Ndim x Nelems x Nqp).

        """
        raise NotImplementedError("!")

    def G(self,
          X: ndarray,
          find: Optional[ndarray] = None) -> ndarray:
        """Perform a mapping from the reference facet to global facet.

        Parameters
        ----------
        X
            Local points on the reference element (Ndim x Nqp).
        find
            A set of facet indices to map to

        Returns
        -------
        ndarray
            Global points (Ndim x Nelems x Nqp).

        """
        raise NotImplementedError("!")

    def detDG(self,
              X: ndarray,
              find: Optional[ndarray] = None) -> ndarray:
        raise NotImplementedError("!")
    
    def normals(self, X, tind, find, t2f):
        raise NotImplementedError("!")

    def detDF(self, X, tind=None):
        raise NotImplementedError("!")

    def DF(self, X, tind=None):
        raise NotImplementedError("!")

    def invDF(self, X, tind=None):
        raise NotImplementedError("!")


class MappingIsoparametric(Mapping):
    """An isoparametric mapping, e.g., for quadrilateral and
    hexahedral elements."""

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 bndelem: Optional[Element] = None):
        """Initialize an isoparametric mapping between
        the reference and the global element.

        This means that the mapping is defined through

        .. math::
            x = F(\widehat{x}) = \sum_{i=1}^N x_i \phi_i(\widehat{x}),
        
        where :math:`N` is the number of basis functions in the provided
        element and :math:`x_i` are the locations of the corresponding
        global nodes.

        Parameters
        ----------
        mesh
            An object of type `~skfem.mesh.Mesh`.
        elem
            An object of type `~skfem.element.Element`.
        bndelem
            An object of type `~skfem.element.Element`.
            Should be a boundary element type corresponding
            to elem, i.e. for ElementHex1 the corresponding
            bndelem is ElementQuad1.

        """
        p = mesh.p
        t = mesh.t
        facets = mesh.facets

        def map(i, X, tind=None):
            if tind is None:
                out = np.zeros((t.shape[1], X.shape[1]))
                for itr in range(t.shape[0]):
                    phi, _ = elem.lbasis(X, itr)
                    out += p[i, t[itr, :]][:, None]*phi
                return out
            else:
                out = np.zeros((len(tind), X.shape[2]))
                for itr in range(t.shape[0]):
                    phi, _ = elem.lbasis(X, itr)
                    out += p[i, t[itr, tind]][:, None]*phi
                return out

        def J(i, j, X, tind=None):
            if tind is None:
                out = np.zeros((t.shape[1], X.shape[1]))
                for itr in range(t.shape[0]):
                    _, dphi = elem.lbasis(X, itr)
                    out += p[i, t[itr, :]][:, None]*dphi[j]
                return out
            else:
                out = np.zeros((len(tind), X.shape[2]))
                for itr in range(t.shape[0]):
                    _, dphi = elem.lbasis(X, itr)
                    out += p[i, t[itr, tind]][:, None]*dphi[j]
                return out

        def bndmap(i, X, find=None):
            if find is None:
                out = np.zeros((facets.shape[1], X.shape[1]))
                for itr in range(facets.shape[0]):
                    phi, _ = bndelem.lbasis(X, itr)
                    out += p[i, facets[itr, :]][:, None]*phi
                return out
            else:
                out = np.zeros((len(find), X.shape[1]))
                for itr in range(facets.shape[0]):
                    phi, _ = bndelem.lbasis(X, itr)
                    out += p[i, facets[itr, find]][:, None]*phi
                return out

        def bndJ(i, j, X, find=None):
            if find is None:
                out = np.zeros((facets.shape[1], X.shape[1]))
                for itr in range(facets.shape[0]):
                    _, dphi = bndelem.lbasis(X, itr)
                    out += p[i, facets[itr, :]][:, None]*dphi[j]
                return out
            else:
                out = np.zeros((len(find), X.shape[1]))
                for itr in range(facets.shape[0]):
                    _, dphi = bndelem.lbasis(X, itr)
                    out += p[i, facets[itr, find]][:, None]*dphi[j]
                return out

        self.map = map
        self.bndmap = bndmap
        self.J = J
        self.bndJ = bndJ
        self.elem = elem
        self.mesh = mesh

    def G(self, X: ndarray, find: Optional[ndarray] = None) -> ndarray:
        return np.array([self.bndmap(i, X, find=find) for i in range(self.mesh.dim())])

    def detDG(self, X: ndarray, find: Optional[ndarray] = None):
        dim = self.mesh.p.shape[0]

        if dim == 2:
            return np.sqrt(self.bndJ(0, 0, X, find)**2 + self.bndJ(1, 0, X, find)**2)
        elif dim == 3:
            return np.sqrt(( self.bndJ(1, 0, X, find)*self.bndJ(2, 1, X, find) -\
                             self.bndJ(2, 0, X, find)*self.bndJ(1, 1, X, find))**2 +
                           (-self.bndJ(0, 0, X, find)*self.bndJ(2, 1, X, find) +\
                             self.bndJ(2, 0, X, find)*self.bndJ(0, 1, X, find))**2 +
                           ( self.bndJ(0, 0, X, find)*self.bndJ(1, 1, X, find) -\
                             self.bndJ(1, 0, X, find)*self.bndJ(0, 1, X, find))**2)
        else:
            raise NotImplementedError("!")

    def invF(self, x, tind=None):
        X = np.zeros(x.shape) + 0.1
        for itr in range(2):
            F = self.F(X, tind)
            invDF = self.invDF(X, tind)
            X = X + np.einsum('ijkl,jkl->ikl', invDF, x - F)
        return X

    def F(self, X, tind=None):
        return np.array([self.map(i, X, tind) for i in range(X.shape[0])])

    def detDF(self, X, tind=None):
        dim = X.shape[0]

        if dim == 2:
            detDF = self.J(0, 0, X, tind=tind) * self.J(1, 1, X, tind=tind) -\
                    self.J(0, 1, X, tind=tind) * self.J(1, 0, X, tind=tind)
        elif dim == 3:
            detDF = self.J(0, 0, X, tind=tind) * (self.J(1, 1, X, tind=tind) * self.J(2, 2, X, tind=tind) -\
                                                  self.J(1, 2, X, tind=tind) * self.J(2, 1, X, tind=tind)) \
                  - self.J(0, 1, X, tind=tind) * (self.J(1, 0, X, tind=tind) * self.J(2, 2, X, tind=tind) -\
                                                  self.J(1, 2, X, tind=tind) * self.J(2, 0, X, tind=tind)) \
                  + self.J(0, 2, X, tind=tind) * (self.J(1, 0, X, tind=tind) * self.J(2, 1, X, tind=tind) -\
                                                  self.J(1, 1, X, tind=tind) * self.J(2, 0, X, tind=tind))
        else:
            raise Exception("Not implemented for the given dimension.")

        if np.sum(detDF==0)>0:
            raise Exception("Zero Jacobian determinant")

        return detDF

    def invDF(self, X, tind=None):
        dim = X.shape[0]
        detDF = self.detDF(X, tind)

        if dim == 2:
            invDF = np.empty((2, 2) + self.J(0, 0, X, tind=tind).shape)
            invDF[0, 0] =  self.J(1, 1, X, tind=tind) / detDF
            invDF[0, 1] = -self.J(0, 1, X, tind=tind) / detDF
            invDF[1, 0] = -self.J(1, 0, X, tind=tind) / detDF
            invDF[1, 1] =  self.J(0, 0, X, tind=tind) / detDF
        elif dim == 3:
            invDF = np.empty((3, 3) + self.J(0, 0, X, tind=tind).shape)
            invDF[0, 0] = (-self.J(1, 2, X, tind=tind) * self.J(2, 1, X, tind=tind) +\
                            self.J(1, 1, X, tind=tind) * self.J(2, 2, X, tind=tind)) / detDF
            invDF[1, 0] = ( self.J(1, 2, X, tind=tind) * self.J(2, 0, X, tind=tind) -\
                            self.J(1, 0, X, tind=tind) * self.J(2, 2, X, tind=tind)) / detDF
            invDF[2, 0] = (-self.J(1, 1, X, tind=tind) * self.J(2, 0, X, tind=tind) +\
                            self.J(1, 0, X, tind=tind) * self.J(2, 1, X, tind=tind)) / detDF
            invDF[0, 1] = ( self.J(0, 2, X, tind=tind) * self.J(2, 1, X, tind=tind) -\
                            self.J(0, 1, X, tind=tind) * self.J(2, 2, X, tind=tind)) / detDF
            invDF[1, 1] = (-self.J(0, 2, X, tind=tind) * self.J(2, 0, X, tind=tind) +\
                            self.J(0, 0, X, tind=tind) * self.J(2, 2, X, tind=tind)) / detDF
            invDF[2, 1] = ( self.J(0, 1, X, tind=tind) * self.J(2, 0, X, tind=tind) -\
                            self.J(0, 0, X, tind=tind) * self.J(2, 1, X, tind=tind)) / detDF
            invDF[0, 2] = (-self.J(0, 2, X, tind=tind) * self.J(1, 1, X, tind=tind) +\
                            self.J(0, 1, X, tind=tind) * self.J(1, 2, X, tind=tind)) / detDF
            invDF[1, 2] = ( self.J(0, 2, X, tind=tind) * self.J(1, 0, X, tind=tind) -\
                            self.J(0, 0, X, tind=tind) * self.J(1, 2, X, tind=tind)) / detDF
            invDF[2, 2] = (-self.J(0, 1, X, tind=tind) * self.J(1, 0, X, tind=tind) +\
                            self.J(0, 0, X, tind=tind) * self.J(1, 1, X, tind=tind)) / detDF
        else:
            raise Exception("Not implemented for the given dimension.")

        return invDF
    
    def normals(self, X, tind, find, t2f):
        # TODO implement this
        return 0


class MappingAffine(Mapping):
    """An affine mapping for simplical elements."""
    def __init__(self, mesh):
        dim = mesh.p.shape[0]

        if mesh.t.shape[0] > 0:
            nt = mesh.t.shape[1]
            # initialize the affine mapping
            self.A = np.empty((dim, dim, nt))
            self.b = np.empty((dim, nt))

            for i in range(dim):
                self.b[i] = mesh.p[i, mesh.t[0, :]]
                for j in range(dim):
                    self.A[i, j] = mesh.p[i, mesh.t[j+1, :]] - mesh.p[i, mesh.t[0, :]]

            # determinants
            if dim == 1:
                self.detA = self.A[0, 0]
            elif dim == 2:
                self.detA = self.A[0, 0] * self.A[1, 1] - self.A[0, 1] * self.A[1, 0]
            elif dim == 3:
                self.detA = self.A[0, 0] * (self.A[1, 1] * self.A[2, 2] - self.A[1, 2] * self.A[2, 1]) -\
                            self.A[0, 1] * (self.A[1, 0] * self.A[2, 2] - self.A[1, 2] * self.A[2, 0]) +\
                            self.A[0, 2] * (self.A[1, 0] * self.A[2, 1] - self.A[1, 1] * self.A[2, 0])
            else:
                raise Exception("Not implemented for the given dimension.")

            # affine mapping inverses
            self.invA = np.empty((dim, dim, nt))
            if dim == 1:
                self.invA[0, 0] = 1.0 / self.A[0, 0]
            elif dim == 2:
                self.invA[0, 0] =  self.A[1, 1] / self.detA
                self.invA[0, 1] = -self.A[0, 1] / self.detA
                self.invA[1, 0] = -self.A[1, 0] / self.detA
                self.invA[1, 1] =  self.A[0, 0] / self.detA
            elif dim == 3:
                self.invA[0, 0] = (-self.A[1, 2] * self.A[2, 1] + self.A[1, 1] * self.A[2, 2]) / self.detA
                self.invA[1, 0] = ( self.A[1, 2] * self.A[2, 0] - self.A[1, 0] * self.A[2, 2]) / self.detA
                self.invA[2, 0] = (-self.A[1, 1] * self.A[2, 0] + self.A[1, 0] * self.A[2, 1]) / self.detA
                self.invA[0, 1] = ( self.A[0, 2] * self.A[2, 1] - self.A[0, 1] * self.A[2, 2]) / self.detA
                self.invA[1, 1] = (-self.A[0, 2] * self.A[2, 0] + self.A[0, 0] * self.A[2, 2]) / self.detA
                self.invA[2, 1] = ( self.A[0, 1] * self.A[2, 0] - self.A[0, 0] * self.A[2, 1]) / self.detA
                self.invA[0, 2] = (-self.A[0, 2] * self.A[1, 1] + self.A[0, 1] * self.A[1, 2]) / self.detA
                self.invA[1, 2] = ( self.A[0, 2] * self.A[1, 0] - self.A[0, 0] * self.A[1, 2]) / self.detA
                self.invA[2, 2] = (-self.A[0, 1] * self.A[1, 0] + self.A[0, 0] * self.A[1, 1]) / self.detA
            else:
                raise Exception("Not implemented for the given dimension.")

        if hasattr(mesh, 'facets'):
            nf = mesh.facets.shape[1]
            # initialize the boundary mapping
            self.B = np.empty((dim, dim-1, nf))
            self.c = np.empty((dim, nf))

            for i in range(dim):
                self.c[i] = mesh.p[i, mesh.facets[0, :]]
                for j in range(dim-1):
                    self.B[i, j] = mesh.p[i, mesh.facets[j+1, :]] - mesh.p[i, mesh.facets[0, :]]

            # area scaling
            if dim == 1:
                self.detB = np.ones(nf)
            elif dim == 2:
                self.detB = np.sqrt(self.B[0, 0]**2 + self.B[1, 0]**2)
            elif dim == 3:
                self.detB = np.sqrt(( self.B[1, 0]*self.B[2, 1] - self.B[2, 0]*self.B[1, 1])**2 +
                                    (-self.B[0, 0]*self.B[2, 1] + self.B[2, 0]*self.B[0, 1])**2 +
                                    ( self.B[0, 0]*self.B[1, 1] - self.B[1, 0]*self.B[0, 1])**2)
            else:
                raise Exception("Not implemented for the given dimension.")

        self.dim = dim
        self.mesh = mesh  # this is required in ElementH2


    def F(self, X, tind=None):
        if tind is None:
            A, b = self.A, self.b
        else:
            A, b = self.A[:, :, tind], self.b[:, tind]

        if len(X.shape) == 2:
            return (np.einsum('ijk,jl', A, X).T + b.T).T
        elif len(X.shape) == 3:
            return (np.einsum('ijk,jkl->ikl', A, X).T + b.T).T
        else:
            raise Exception("Wrong dimension of input.")

    def invF(self, x, tind=None):
        if tind is None:
            invA, b = self.invA, self.b
        else:
            invA, b = self.invA[:, :, tind], self.b[:, tind]

        y = (x.T - b.T).T

        return np.einsum('ijk,jkl->ikl', invA, y)

    def detDF(self, X, tind=None):
        if tind is None:
            detDF = self.detA
        else:
            detDF = self.detA[tind]

        return np.tile(detDF, (X.shape[1], 1)).T

    def DF(self, X, tind=None):
        if tind is None:
            DF = self.A
        else:
            DF = self.A[:, :, tind]

        return np.einsum('ijk,l->ijkl', DF, 1 + 0*X[0, :])

    def invDF(self, X, tind=None):
        if tind is None:
            invDF = self.invA
        else:
            invDF = self.invA[:, :, tind]

        ones = np.ones(X.shape[-1])
        return np.einsum('ijk,l->ijkl', invDF, ones)

    def G(self, X, find=None):
        if find is None:
            B, c = self.B, self.c
        else:
            B, c = self.B[:, :, find], self.c[:, find]

        return (np.einsum('ijk,jl', B, X).T + c.T).T

    def detDG(self, X, find=None):
        if find is None:
            detDG = self.detB
        else:
            detDG = self.detB[find]

        return np.tile(detDG, (X.shape[1], 1)).T

    def normals(self, X, tind, find, t2f):
        if self.dim == 1:
            Nref = np.array([[-1.0],
                             [ 1.0]])
        elif self.dim == 2:
            Nref = np.array([[0.0, -1.0],
                             [1.0, 1.0],
                             [-1.0, 0.0]])
        elif self.dim == 3:
            Nref = np.array([[0.0, 0.0, -1.0],
                             [0.0, -1.0, 0.0],
                             [-1.0, 0.0, 0.0],
                             [1.0, 1.0, 1.0]])
        else:
            raise Exception("Not implemented for the given dimension.")

        invDF = self.invDF(X, tind)
        N = np.empty((self.dim, len(find)))

        for itr in range(Nref.shape[0]):
            ix = np.nonzero(t2f[itr, tind] == find)[0]
            for jtr in range(Nref.shape[1]):
                N[jtr, ix] = Nref[itr, jtr]

        n = np.einsum('ijkl,ik->jkl', invDF, N)
        nlength = np.sqrt(np.sum(n**2, axis=0))
        return np.einsum('ijk,jk->ijk', n, 1.0/nlength)

