import numpy as np
from numpy.polynomial.legendre import leggauss, legval
from scipy.special import eval_jacobi

from ..element_hdiv import ElementHdiv
from ...refdom import RefTri
from ...quadrature import get_quadrature
from .element_tri_p import _scaled_legendre


def _dubiner(X, n):
    """Evaluate the Dubiner polynomials of degree at most n.

    Returns the values, the partial derivatives and the degrees.  The
    polynomials are orthogonal over the reference triangle.

    """
    x, y = X
    s, t = 2. * x + y - 1., 1. - y
    A, As, At = _scaled_legendre(n, s, t)
    vals, dxs, dys, degs = [], [], [], []
    for d in range(n + 1):
        for i in range(d, -1, -1):
            j = d - i
            B = eval_jacobi(j, 2 * i + 1, 0, 2. * y - 1.)
            dB = (0. * y if j == 0 else
                  (j + 2 * i + 2) * eval_jacobi(j - 1, 2 * i + 2, 1,
                                                2. * y - 1.))
            vals.append(A[i] * B)
            dxs.append(2. * As[i] * B)
            dys.append((As[i] - At[i]) * B + A[i] * dB)
            degs.append(d)
    return np.array(vals), np.array(dxs), np.array(dys), np.array(degs)


class ElementTriRT(ElementHdiv):
    """The Raviart-Thomas element of degree :math:`p`.

    The DOFs are the moments of the normal component against the Legendre
    polynomials of degree :math:`p - 1` on the facets, and the moments
    against the vector valued polynomials of degree :math:`p - 2` in the
    interior.  ``ElementTriRT(1)`` is the lowest order element.

    """

    refdom = RefTri

    def __init__(self, p):
        if p < 1:
            raise ValueError("p < 1 not supported.")
        self.p = p
        self.maxdeg = p
        self.facet_dofs = p
        self.interior_dofs = p * (p - 1)
        self.dofnames = ['u^n'] * p + ['NA'] * self.interior_dofs
        s = (np.arange(p) + 1.) / (p + 1.)
        P = self.refdom.p
        self.doflocs = np.array(
            [P[:, a] + sj * (P[:, b] - P[:, a])
             for a, b in self.refdom.facets
             for sj in s]
            + [[1. / 3., 1. / 3.]] * self.interior_dofs
        )
        # normalize the Dubiner polynomials in L2
        Xq, Wq = get_quadrature(self.refdom, 2 * p)
        self._scale = 1. / np.sqrt(np.sum(_dubiner(Xq, p - 1)[0] ** 2 * Wq,
                                          axis=1))
        self._coeffs = np.linalg.inv(self._dof_matrix()).T
        self._X = np.array([])

    def _eval_space(self, X):
        """Evaluate a basis of the polynomial space and its divergence."""
        v, vx, vy, deg = _dubiner(X, self.p - 1)
        v, vx, vy = (self._scale[(slice(None),) + (None,) * (X.ndim - 1)] * f
                     for f in (v, vx, vy))
        zero = 0. * v[0]
        phi, div = [], []
        for k in range(len(v)):
            phi += [np.array([v[k], zero]), np.array([zero, v[k]])]
            div += [vx[k], vy[k]]
        x, y = X
        for k in np.nonzero(deg == self.p - 1)[0]:
            phi.append(np.array([x * v[k], y * v[k]]))
            div.append(2. * v[k] + x * vx[k] + y * vy[k])
        return np.array(phi), np.array(div)

    def _dof_matrix(self):
        """Evaluate the DOFs for the basis of the polynomial space."""
        rows = []
        # normal moments on the facets
        s, w = leggauss(self.p + 1)
        s, w = (s + 1.) / 2., w / 2.
        P = self.refdom.p
        for a, b in self.refdom.facets:
            tangent = P[:, b] - P[:, a]
            length = np.linalg.norm(tangent)
            normal = np.array([tangent[1], -tangent[0]]) / length
            if np.dot(normal, P[:, a] - P.mean(axis=1)) < 0:
                normal = -normal
            phi, _ = self._eval_space(P[:, [a]] + tangent[:, None] * s)
            un = np.einsum('i,kiq->kq', normal, phi)
            for j in range(self.p):
                c = np.zeros(j + 1)
                c[j] = 1.
                rows.append(np.sum(un * legval(2. * s - 1., c) * w,
                                   axis=1) * length)
        # moments against vector valued polynomials in the interior
        if self.p > 1:
            X, W = get_quadrature(self.refdom, 2 * self.p)
            phi, _ = self._eval_space(X)
            q = _dubiner(X, self.p - 2)[0]
            for m in range(len(q)):
                for c in range(2):
                    rows.append(np.sum(phi[:, c] * q[m] * W, axis=1))
        return np.array(rows)

    def reversed_facet_dof(self, j):
        # P_j(-s) = (-1) ** j P_j(s)
        return j, (-1.) ** j

    def lbasis(self, X, i):
        if i < 0 or i >= 3 * self.facet_dofs + self.interior_dofs:
            self._index_error()
        if self._X.shape != X.shape or (self._X != X).any():
            self._X = X.copy()
            self._space = self._eval_space(X)
        phi, div = self._space
        return (np.einsum('k,k...->...', self._coeffs[i], phi),
                np.einsum('k,k...->...', self._coeffs[i], div))
