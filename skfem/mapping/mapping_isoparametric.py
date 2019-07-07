import numpy as np

from typing import Optional
from numpy import ndarray

from skfem.mesh import Mesh
from skfem.element import Element
from .mapping import Mapping


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
                out = np.zeros((len(tind), X.shape[-1]))
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
                out = np.zeros((len(tind), X.shape[-1]))
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
        self.dim = mesh.p.shape[0]

    def G(self, X: ndarray, find: Optional[ndarray] = None) -> ndarray:
        return np.array([self.bndmap(i, X, find=find)
                         for i in range(self.mesh.dim())])

    def detDG(self, X: ndarray, find: Optional[ndarray] = None):
        if self.dim == 2:
            return np.sqrt(self.bndJ(0, 0, X, find)**2 +
                           self.bndJ(1, 0, X, find)**2)
        elif self.dim == 3:
            return np.sqrt(
                (self.bndJ(1, 0, X, find) * self.bndJ(2, 1, X, find) -
                 self.bndJ(2, 0, X, find) * self.bndJ(1, 1, X, find))**2 +
                (-self.bndJ(0, 0, X, find) * self.bndJ(2, 1, X, find) +
                 self.bndJ(2, 0, X, find) * self.bndJ(0, 1, X, find))**2 +
                (self.bndJ(0, 0, X, find) * self.bndJ(1, 1, X, find) -
                 self.bndJ(1, 0, X, find) * self.bndJ(0, 1, X, find))**2
            )
        else:
            raise NotImplementedError("!")

    def invF(self, x, tind=None):
        """Newton iteration for evaluating inverse isoparametric mapping."""
        X = np.zeros(x.shape)
        for itr in range(50):
            F = self.F(X, tind)
            invDF = self.invDF(X, tind)
            dX = np.einsum('ijkl,jkl->ikl', invDF, x - F)
            X = X + dX
            if np.sum(dX) < 1e-6:
                 break
        if (np.abs(X) > 1.0).any():
            raise ValueError("Inverse mapped point outside reference element!")
        return X

    def F(self, X, tind=None):
        return np.array([self.map(i, X, tind) for i in range(X.shape[0])])

    def detDF(self, X, tind=None):
        if self.dim == 2:
            detDF = (self.J(0, 0, X, tind=tind) * self.J(1, 1, X, tind=tind) -
                     self.J(0, 1, X, tind=tind) * self.J(1, 0, X, tind=tind))
        elif self.dim == 3:
            detDF = (self.J(0, 0, X, tind=tind) *\
                     (self.J(1, 1, X, tind=tind) * self.J(2, 2, X, tind=tind) -
                      self.J(1, 2, X, tind=tind) * self.J(2, 1, X, tind=tind))
                     - self.J(0, 1, X, tind=tind) *\
                     (self.J(1, 0, X, tind=tind) * self.J(2, 2, X, tind=tind) -
                      self.J(1, 2, X, tind=tind) * self.J(2, 0, X, tind=tind))
                     + self.J(0, 2, X, tind=tind) *\
                     (self.J(1, 0, X, tind=tind) * self.J(2, 1, X, tind=tind) -
                      self.J(1, 1, X, tind=tind) * self.J(2, 0, X, tind=tind)))
        else:
            raise Exception("Not implemented for the given dimension.")

        if np.sum(detDF==0)>0:
            raise Exception("Zero Jacobian determinant")

        return detDF

    def invDF(self, X, tind=None):
        detDF = self.detDF(X, tind)

        if self.dim == 2:
            invDF = np.empty((2, 2) + self.J(0, 0, X, tind=tind).shape)
            invDF[0, 0] =  self.J(1, 1, X, tind=tind) / detDF
            invDF[0, 1] = -self.J(0, 1, X, tind=tind) / detDF
            invDF[1, 0] = -self.J(1, 0, X, tind=tind) / detDF
            invDF[1, 1] =  self.J(0, 0, X, tind=tind) / detDF
        elif self.dim == 3:
            invDF = np.empty((3, 3) + self.J(0, 0, X, tind=tind).shape)
            invDF[0, 0] = (-self.J(1, 2, X, tind=tind) *\
                           self.J(2, 1, X, tind=tind) +
                           self.J(1, 1, X, tind=tind) *\
                           self.J(2, 2, X, tind=tind)) / detDF
            invDF[1, 0] = (self.J(1, 2, X, tind=tind) *\
                           self.J(2, 0, X, tind=tind) -
                           self.J(1, 0, X, tind=tind) *\
                           self.J(2, 2, X, tind=tind)) / detDF
            invDF[2, 0] = (-self.J(1, 1, X, tind=tind) *\
                           self.J(2, 0, X, tind=tind) +
                           self.J(1, 0, X, tind=tind) *\
                           self.J(2, 1, X, tind=tind)) / detDF
            invDF[0, 1] = (self.J(0, 2, X, tind=tind) *\
                           self.J(2, 1, X, tind=tind) +
                           -self.J(0, 1, X, tind=tind) *\
                           self.J(2, 2, X, tind=tind)) / detDF
            invDF[1, 1] = (-self.J(0, 2, X, tind=tind) *\
                           self.J(2, 0, X, tind=tind) +
                           self.J(0, 0, X, tind=tind) *\
                           self.J(2, 2, X, tind=tind)) / detDF
            invDF[2, 1] = (self.J(0, 1, X, tind=tind) *\
                           self.J(2, 0, X, tind=tind) -
                           self.J(0, 0, X, tind=tind) *\
                           self.J(2, 1, X, tind=tind)) / detDF
            invDF[0, 2] = (-self.J(0, 2, X, tind=tind) *\
                           self.J(1, 1, X, tind=tind) +
                           self.J(0, 1, X, tind=tind) *\
                           self.J(1, 2, X, tind=tind)) / detDF
            invDF[1, 2] = (self.J(0, 2, X, tind=tind) *\
                           self.J(1, 0, X, tind=tind) -
                           self.J(0, 0, X, tind=tind) *\
                           self.J(1, 2, X, tind=tind)) / detDF
            invDF[2, 2] = (-self.J(0, 1, X, tind=tind) *\
                           self.J(1, 0, X, tind=tind) +
                           self.J(0, 0, X, tind=tind) *\
                           self.J(1, 1, X, tind=tind)) / detDF
        else:
            raise Exception("Not implemented for the given dimension.")

        return invDF
    
    def normals(self, X, tind, find, t2f):
        if self.dim == 1:
            Nref = np.array([[-1.0],
                             [ 1.0]])
        elif self.dim == 2:
            Nref = np.array([[0.0, -1.0],
                             [1.0, 0.0],
                             [0.0, 1.0],
                             [-1.0, 0.0]])
        elif self.dim == 3:
            Nref = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0],
                             [0.0, -1.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [-1.0, 0.0, 0.0]])
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
