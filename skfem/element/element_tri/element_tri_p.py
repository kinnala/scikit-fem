import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTri


def _scaled_legendre(n, s, t):
    """Evaluate the scaled Legendre polynomials t ** k P_k(s / t), k <= n.

    Returns the values and the derivatives with respect to s and t.

    """
    P = np.zeros((n + 1,) + s.shape)
    Ps = np.zeros_like(P)
    Pt = np.zeros_like(P)
    P[0] = 1.
    if n > 0:
        P[1] = s
        Ps[1] = 1.
    for k in range(1, n):
        P[k + 1] = ((2 * k + 1) * s * P[k] - k * t ** 2 * P[k - 1]) / (k + 1)
        Ps[k + 1] = ((2 * k + 1) * (P[k] + s * Ps[k])
                     - k * t ** 2 * Ps[k - 1]) / (k + 1)
        Pt[k + 1] = ((2 * k + 1) * s * Pt[k]
                     - k * (2 * t * P[k - 1] + t ** 2 * Pt[k - 1])) / (k + 1)
    return P, Ps, Pt


def _scaled_integrated_legendre(n, s, t):
    """Evaluate t ** k L_k(s / t), k <= n, with L_k the integral of P_{k-1}.

    Returns the values and the derivatives with respect to s and t.  The
    values for k < 2 are not used.

    """
    P, Ps, Pt = _scaled_legendre(n, s, t)
    L = np.zeros_like(P)
    Ls = np.zeros_like(P)
    Lt = np.zeros_like(P)
    for k in range(2, n + 1):
        L[k] = (P[k] - t ** 2 * P[k - 2]) / (2 * k - 1)
        Ls[k] = (Ps[k] - t ** 2 * Ps[k - 2]) / (2 * k - 1)
        Lt[k] = (Pt[k] - 2 * t * P[k - 2] - t ** 2 * Pt[k - 2]) / (2 * k - 1)
    return L, Ls, Lt


class ElementTriP(ElementH1):
    """Piecewise :math:`p`'th order element with a hierarchical basis.

    The vertex basis functions are the barycentric coordinates, the edge
    basis functions are scaled integrated Legendre polynomials and the
    interior basis functions are bubbles multiplied by products of scaled
    Legendre polynomials.  The basis of degree :math:`p` is contained in the
    basis of degree :math:`p + 1`.  Only the vertex DOFs are nodal.

    """

    nodal_dofs = 1
    refdom = RefTri

    def __init__(self, p):
        if p < 1:
            raise ValueError("p < 1 not supported.")
        self.p = p
        self.maxdeg = p
        self.facet_dofs = p - 1
        self.interior_dofs = (p - 1) * (p - 2) // 2
        self.dofnames = ['u'] * (1 + self.facet_dofs + self.interior_dofs)
        self.doflocs = np.array(
            [[0., 0.], [1., 0.], [0., 1.]]
            + [[np.nan, np.nan]] * (3 * self.facet_dofs + self.interior_dofs)
        )
        self._interior = [(i, n - i)
                          for n in range(p - 2)
                          for i in range(n, -1, -1)]

    def reversed_facet_dof(self, j):
        # L_k(-s) = (-1) ** k L_k(s) where k = j + 2
        return j, (-1.) ** j

    def lbasis(self, X, i):
        x, y = X
        lam = np.array([1. - x - y, x, y])
        dlam = np.array([[-1., -1.], [1., 0.], [0., 1.]])
        zero = 0. * x

        if i < 3:
            phi = lam[i]
            dphi = np.array([zero + dlam[i, 0], zero + dlam[i, 1]])

        elif i < 3 + 3 * self.facet_dofs:
            facet, j = divmod(i - 3, self.facet_dofs)
            a, b = self.refdom.facets[facet]
            k = j + 2
            L, Ls, Lt = _scaled_integrated_legendre(k, lam[b] - lam[a],
                                                    lam[a] + lam[b])
            phi = L[k]
            dphi = np.array([Ls[k] * (dlam[b, d] - dlam[a, d])
                             + Lt[k] * (dlam[a, d] + dlam[b, d])
                             for d in range(2)])

        elif i < 3 + 3 * self.facet_dofs + self.interior_dofs:
            m, n = self._interior[i - 3 - 3 * self.facet_dofs]
            bubble = lam[0] * lam[1] * lam[2]
            dbubble = np.array([
                sum(dlam[c, d] * np.prod(np.delete(lam, c, axis=0), axis=0)
                    for c in range(3))
                for d in range(2)
            ])
            P, Ps, Pt = _scaled_legendre(m, lam[1] - lam[0], lam[0] + lam[1])
            Q, Qs, _ = _scaled_legendre(n, 2. * lam[2] - 1., 1. + zero)
            f, g = P[m], Q[n]
            df = np.array([Ps[m] * (dlam[1, d] - dlam[0, d])
                           + Pt[m] * (dlam[0, d] + dlam[1, d])
                           for d in range(2)])
            dg = np.array([2. * Qs[n] * dlam[2, d] for d in range(2)])
            phi = bubble * f * g
            dphi = dbubble * f * g + bubble * df * g + bubble * f * dg

        else:
            self._index_error()

        return phi, dphi
