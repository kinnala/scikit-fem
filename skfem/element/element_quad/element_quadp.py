import numpy as np

from ...refdom import RefQuad
from ..element_line import ElementLinePp


class ElementQuadP(ElementLinePp):

    nodal_dofs = 1
    refdom = RefQuad

    def __init__(self, p):

        self.facet_dofs = p - 1
        self.interior_dofs = (p + 1) ** 2 - 4 * self.facet_dofs - 4
        self.maxdeg = p ** 2
        self.dofnames = (1 + self.facet_dofs + self.interior_dofs) * ['u']
        N = 4 * self.facet_dofs + self.interior_dofs
        self.doflocs = np.array([[0., 0.],
                                 [1., 0.],
                                 [1., 1.],
                                 [0., 1.]] +
                                [[np.nan, np.nan] for i in range(N)])
        self.Px, self.Py = np.zeros((0, 0)), np.zeros((0, 0))
        self.dPx, self.dPy = np.zeros((0, 0, 1)), np.zeros((0, 0, 1))
        self.p = p

    def lbasis(self, X, i):
        x, y = X

        if self.Px.shape[1] != len(x):
            self.Px, self.dPx = self._reval_legendre(x, self.p)
            self.Py, self.dPy = self._reval_legendre(y, self.p)

        if i < 4:
            order = [(0, 0), (1, 0), (1, 1), (0, 1)]
            Px, Py = self.Px[order[i][0]], self.Py[order[i][1]]
            dPx, dPy = self.dPx[order[i][0]], self.dPy[order[i][1]]

        elif i < 4 + 4 * self.facet_dofs:
            ind = ((i - 4) % self.facet_dofs) + 2
            n = (i - 4) // self.facet_dofs
            if n == 0:
                Px, Py = self.Px[ind], self.Py[0]
                dPx, dPy = self.dPx[ind], self.dPy[0]
            elif n == 1:
                Px, Py = self.Px[1], self.Py[ind]
                dPx, dPy = self.dPx[1], self.dPy[ind]
            elif n == 2:
                Px, Py = self.Px[ind], self.Py[1]
                dPx, dPy = self.dPx[ind], self.dPy[1]
            elif n == 3:
                Px, Py = self.Px[0], self.Py[ind]
                dPx, dPy = self.dPx[0], self.dPy[ind]

        else:
            # go through rest of the dofs in arbitrary order
            j = i - 4 - 4 * self.facet_dofs
            ix = (j // (self.p - 1)) + 2
            iy = (j % (self.p - 1)) + 2
            Px, Py = self.Px[ix], self.Py[iy]
            dPx, dPy = self.dPx[ix], self.dPy[iy]

        return Px * Py, np.array([dPx[0] * Py, dPy[0] * Px])
