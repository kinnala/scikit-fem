from typing import Optional, Union

import numpy as np
from numpy import ndarray

from skfem.element import Element
from skfem.mesh import Mesh2D, Mesh3D
from .mapping import Mapping
from ..generic_utils import hash_args


class MappingIsoparametric(Mapping):
    """An isoparametric mapping, e.g., for quadrilateral and
    hexahedral elements."""

    def __init__(self,
                 mesh: Union[Mesh2D, Mesh3D],
                 elem: Element,
                 bndelem: Optional[Element] = None):
        r"""Initialize an isoparametric mapping between
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
        self.elem = elem
        self.bndelem = bndelem
        self.mesh = mesh
        self.dim = mesh.dim()

    def Fmap(self, i, X, tind=None):
        p = self.mesh.doflocs
        t = self.mesh.dofs.element_dofs
        if tind is None:
            out = np.zeros((t.shape[1], X.shape[1]))
            for itr in range(t.shape[0]):
                phi, _ = self.elem.lbasis(X, itr)
                out += p[i, t[itr, :]][:, None] * phi
            return out
        else:
            out = np.zeros((len(tind), X.shape[-1]))
            for itr in range(t.shape[0]):
                phi, _ = self.elem.lbasis(X, itr)
                out += p[i, t[itr, tind]][:, None] * phi
            return out

    def bndmap(self, i, X, find=None):
        p = self.mesh.doflocs
        facets = self.mesh.facets
        if find is None:
            out = np.zeros((facets.shape[1], X.shape[1]))
            for itr in range(facets.shape[0]):
                phi, _ = self.bndelem.lbasis(X, itr)
                out += p[i, facets[itr, :]][:, None] * phi
            return out
        else:
            out = np.zeros((len(find), X.shape[-1]))
            for itr in range(facets.shape[0]):
                phi, _ = self.bndelem.lbasis(X, itr)
                out += p[i, facets[itr, find]][:, None] * phi
            return out

    def bndJ(self, i, j, X, find=None):
        p = self.mesh.doflocs
        facets = self.mesh.facets
        if find is None:
            out = np.zeros((facets.shape[1], X.shape[1]))
            for itr in range(facets.shape[0]):
                _, dphi = self.bndelem.lbasis(X, itr)
                out += p[i, facets[itr, :]][:, None] * dphi[j]
            return out
        else:
            out = np.zeros((len(find), X.shape[-1]))
            for itr in range(facets.shape[0]):
                _, dphi = self.bndelem.lbasis(X, itr)
                out += p[i, facets[itr, find]][:, None] * dphi[j]
            return out

    def _J(self, i, j, X, tind=None):
        p = self.mesh.doflocs
        t = self.mesh.dofs.element_dofs
        if tind is None:
            out = np.zeros((t.shape[1], X.shape[1]))
            for itr in range(t.shape[0]):
                _, dphi = self.elem.lbasis(X, itr)
                out += p[i, t[itr, :]][:, None] * dphi[j]
            return out
        else:
            out = np.zeros((len(tind), X.shape[-1]))
            for itr in range(t.shape[0]):
                _, dphi = self.elem.lbasis(X, itr)
                out += p[i, t[itr, tind]][:, None] * dphi[j]
            return out

    def J(self, i, j, X, tind=None):
        # cache the jacobian
        if not hasattr(self, '_cache'):
            self._cache = {}
        h = hash_args(i, j, X, tind)
        if h not in self._cache:
            self._cache[h] = self._J(i, j, X, tind)
        return self._cache[h]

    def G(self, X: ndarray, find: Optional[ndarray] = None) -> ndarray:
        return np.array([self.bndmap(i, X, find=find)
                         for i in range(self.mesh.dim())])

    def detDG(self, X: ndarray, find: Optional[ndarray] = None):
        if self.dim == 2:
            return np.sqrt(self.bndJ(0, 0, X, find) ** 2 +
                           self.bndJ(1, 0, X, find) ** 2)
        elif self.dim == 3:
            return np.sqrt(
                (self.bndJ(1, 0, X, find) * self.bndJ(2, 1, X, find) -
                 self.bndJ(2, 0, X, find) * self.bndJ(1, 1, X, find)) ** 2 +
                (-self.bndJ(0, 0, X, find) * self.bndJ(2, 1, X, find) +
                 self.bndJ(2, 0, X, find) * self.bndJ(0, 1, X, find)) ** 2 +
                (self.bndJ(0, 0, X, find) * self.bndJ(1, 1, X, find) -
                 self.bndJ(1, 0, X, find) * self.bndJ(0, 1, X, find)) ** 2
            )
        else:
            raise NotImplementedError

    def invF(self, x, tind=None, newton_max_iters=50, newton_tol=1e-12):
        """Newton iteration for evaluating inverse isoparametric mapping."""
        X = np.zeros(x.shape) + .5
        for _ in range(newton_max_iters):
            F = self.F(X, tind)
            invDF = self.invDF(X, tind)
            dX = np.einsum('ijkl,jkl->ikl', invDF, x - F)
            X = np.clip(X + dX, 0., 1.)
            if (np.linalg.norm(dX, 1, (0, 2)) < newton_tol).all():
                return X
        raise Exception(("Newton iteration didn't converge "
                         "up to TOL={}".format(newton_tol)))

    def F(self, X, tind=None):
        return np.array([self.Fmap(i, X, tind) for i in range(X.shape[0])])

    def detDF(self, X, tind=None, J=None):
        if J is None:
            J = [[self.J(i, j, X, tind=tind) for j in range(self.dim)]
                 for i in range(self.dim)]

        if self.dim == 1:
            detDF = J[0][0]
        elif self.dim == 2:
            detDF = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        elif self.dim == 3:
            detDF = (J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                     J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                     J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]))
        else:
            raise Exception("Not implemented for the given dimension.")

        if np.sum(detDF == 0) > 0:
            raise Exception("Zero Jacobian determinant")

        return detDF

    def invDF(self, X, tind=None):
        J = [[self.J(i, j, X, tind=tind) for j in range(self.dim)]
             for i in range(self.dim)]
        detDF = self.detDF(X, tind, J=J)
        invDF = np.empty((self.dim, self.dim) + J[0][0].shape)

        if self.dim == 1:
            invDF[0, 0] = np.ones(J[0][0].shape)
        elif self.dim == 2:
            invDF[0, 0] =  J[1][1]  # noqa
            invDF[0, 1] = -J[0][1]
            invDF[1, 0] = -J[1][0]
            invDF[1, 1] =  J[0][0]  # noqa
        elif self.dim == 3:
            invDF[0, 0] = -J[1][2] * J[2][1] + J[1][1] * J[2][2]
            invDF[1, 0] =  J[1][2] * J[2][0] - J[1][0] * J[2][2]  # noqa
            invDF[2, 0] = -J[1][1] * J[2][0] + J[1][0] * J[2][1]
            invDF[0, 1] =  J[0][2] * J[2][1] - J[0][1] * J[2][2]  # noqa
            invDF[1, 1] = -J[0][2] * J[2][0] + J[0][0] * J[2][2]
            invDF[2, 1] =  J[0][1] * J[2][0] - J[0][0] * J[2][1]  # noqa
            invDF[0, 2] = -J[0][2] * J[1][1] + J[0][1] * J[1][2]
            invDF[1, 2] =  J[0][2] * J[1][0] - J[0][0] * J[1][2]  # noqa
            invDF[2, 2] = -J[0][1] * J[1][0] + J[0][0] * J[1][1]
        else:
            raise Exception("Not implemented for the given dimension.")

        return invDF / detDF

    def normals(self, X, tind, find, t2f):
        Nref = self.mesh.elem.refdom.normals

        invDF = self.invDF(X, tind)
        N = np.empty((self.dim, len(find)))

        for itr in range(Nref.shape[0]):
            ix = np.nonzero(t2f[itr, tind] == find)[0]
            for jtr in range(Nref.shape[1]):
                N[jtr, ix] = Nref[itr, jtr]

        n = np.einsum('ijkl,ik->jkl', invDF, N)
        nlength = np.sqrt(np.sum(n ** 2, axis=0))
        return np.einsum('ijk,jk->ijk', n, 1. / nlength)
