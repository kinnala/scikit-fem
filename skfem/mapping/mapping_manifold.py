from typing import Optional

import numpy as np
from numpy import ndarray

from .mapping_isoparametric import MappingIsoparametric


class MappingManifold(MappingIsoparametric):
    r"""An isoparametric mapping for meshes embedded in a higher dimension.

    Used for, e.g., curves in 2D/3D (``MeshLine1`` with a ``2 x N`` or
    ``3 x N`` point array) and surfaces in 3D (``MeshTri1``/``MeshQuad1``
    with a ``3 x N`` point array).

    The Jacobian :math:`J = DF` of the mapping is then a nonsquare
    :math:`D \times d` matrix.  It is replaced in the usual formulae by

    - the area element :math:`\sqrt{\det(J^T J)}` instead of :math:`\det J`,
    - the pseudoinverse :math:`(J^T J)^{-1} J^T` instead of :math:`J^{-1}`.

    Consequently, the gradients computed by
    :class:`~skfem.element.ElementH1` are the tangential (surface) gradients
    with :math:`D` components.

    """

    def __init__(self, mesh, elem, bndelem=None):
        super().__init__(mesh, elem, bndelem)
        self.dim = mesh.dim()  # topological dimension, d
        self.gdim = mesh.doflocs.shape[0]  # geometric dimension, D

    def _jacobian(self, X, tind=None):
        return np.array([[self.J(i, j, X, tind=tind)
                          for j in range(self.dim)]
                         for i in range(self.gdim)])

    def F(self, X, tind=None):
        return np.array([self.Fmap(i, X, tind) for i in range(self.gdim)])

    def DF(self, X, tind=None):
        return self._jacobian(X, tind)

    def detDF(self, X, tind=None):
        J = self._jacobian(X, tind)
        g = np.einsum('kiel,kjel->ijel', J, J)
        return np.sqrt(np.linalg.det(np.moveaxis(g, (0, 1), (-2, -1))))

    def invDF(self, X, tind=None):
        J = self._jacobian(X, tind)
        g = np.einsum('kiel,kjel->ijel', J, J)
        ginv = np.moveaxis(np.linalg.inv(np.moveaxis(g, (0, 1), (-2, -1))),
                           (-2, -1), (0, 1))
        # shape (d, D, nelems, nqp) as in the square case
        return np.einsum('ijel,kjel->ikel', ginv, J)

    def invF(self, x, tind=None, newton_max_iters=50, newton_tol=1e-12):
        """Gauss-Newton iteration; assumes that x lies on the manifold."""
        X = np.zeros((self.dim,) + x.shape[1:]) + .5 / self.dim
        for _ in range(newton_max_iters):
            F = self.F(X, tind)
            invDF = self.invDF(X, tind)
            dX = np.einsum('ijkl,jkl->ikl', invDF, x - F)
            X = X + dX
            if (np.linalg.norm(dX, 1, (0, 2)) < newton_tol).all():
                return X
        raise Exception(("Newton iteration didn't converge "
                         "up to TOL={}".format(newton_tol)))

    def G(self, X: ndarray, find: Optional[ndarray] = None) -> ndarray:
        if self.dim == 1:
            # facets are points
            facets = self.mesh.facets[0]
            if find is not None:
                facets = facets[find]
            return np.repeat(self.mesh.doflocs[:, facets, None],
                             X.shape[-1], axis=2)
        return np.array([self.bndmap(i, X, find=find)
                         for i in range(self.gdim)])

    def detDG(self, X: ndarray, find: Optional[ndarray] = None):
        if self.dim == 1:
            nf = (self.mesh.nfacets if find is None else len(find))
            return np.ones((nf, X.shape[-1]))
        B = np.array([[self.bndJ(i, j, X, find)
                       for j in range(self.dim - 1)]
                      for i in range(self.gdim)])
        g = np.einsum('kiel,kjel->ijel', B, B)
        return np.sqrt(np.linalg.det(np.moveaxis(g, (0, 1), (-2, -1))))

    def tangents(self, X, tind=None):
        """Unit tangent vector of a curve; shape (D, nelems, nqp)."""
        if self.dim != 1:
            raise NotImplementedError
        J = self._jacobian(X, tind)[:, 0]
        return J / np.linalg.norm(J, axis=0)

    def cell_normals(self, X, tind=None):
        """Unit normal of a codimension one manifold; shape (D, nelems, nqp).

        Orientation follows the element node ordering.

        """
        J = self._jacobian(X, tind)
        if self.gdim == 2 and self.dim == 1:
            n = np.array([J[1, 0], -J[0, 0]])
        elif self.gdim == 3 and self.dim == 2:
            n = np.cross(J[:, 0], J[:, 1], axis=0)
        else:
            raise NotImplementedError
        return n / np.linalg.norm(n, axis=0)
