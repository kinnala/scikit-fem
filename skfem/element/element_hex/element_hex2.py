import numpy as np
from numpy.linalg import inv

from ..element_h1 import ElementH1
from ...refdom import RefHex


class ElementHex2(ElementH1):

    nodal_dofs = 1
    facet_dofs = 1
    edge_dofs = 1
    interior_dofs = 1
    maxdeg = 6
    dofnames = ['u', 'u', 'u', 'u']
    doflocs = np.array([[1., 1., 1.],
                        [1., 1., 0.],
                        [1., 0., 1.],
                        [0., 1., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [0., 0., 0.],
                        [1., 1., .5],  # edges
                        [1., .5, 1.],
                        [.5, 1., 1.],
                        [1., .5, 0.],
                        [.5, 1., 0.],
                        [1., 0., .5],
                        [.5, 0., 1.],
                        [0., 1., .5],
                        [0., .5, 1.],
                        [.5, 0., 0.],
                        [0., .5, 0.],
                        [0., 0., .5],
                        [1., .5, .5],  # facets
                        [.5, .5, 1.],
                        [.5, 1., .5],
                        [.5, .0, .5],
                        [.5, .5, 0.],
                        [0., .5, .5],
                        [.5, .5, .5]])
    refdom = RefHex

    def __init__(self):
        X = self.doflocs.T
        V = self._power_basis(X)
        self.invV = inv(V)
        self.invV[np.abs(self.invV) < 1e-10] = 0.

    def _power_basis(self, X):
        return np.array([
            1. + 0. * X[0],
            X[0],
            X[1],
            X[2],
            X[0] ** 2,
            X[1] ** 2,
            X[2] ** 2,
            X[0] * X[1],
            X[1] * X[2],
            X[0] * X[2],
            X[0] ** 2 * X[1],
            X[1] ** 2 * X[2],
            X[0] ** 2 * X[2],
            X[0] * X[1] ** 2,
            X[1] * X[2] ** 2,
            X[0] * X[2] ** 2,
            X[0] ** 2 * X[1] ** 2,
            X[1] ** 2 * X[2] ** 2,
            X[0] ** 2 * X[2] ** 2,
            X[0] * X[1] * X[2],
            X[0] ** 2 * X[1] * X[2],
            X[0] * X[1] ** 2 * X[2],
            X[0] * X[1] * X[2] ** 2,
            X[0] ** 2 * X[1] ** 2 * X[2],
            X[0] * X[1] ** 2 * X[2] ** 2,
            X[0] ** 2 * X[1] * X[2] ** 2,
            X[0] ** 2 * X[1] ** 2 * X[2] ** 2,
        ])

    def _power_basis_dx(self, X):
        return np.array([
            0. * X[0],
            1. + 0. * X[0],
            0. * X[0],
            0. * X[0],
            2. * X[0],
            0. * X[0],
            0. * X[0],
            X[1],
            0. * X[0],
            X[2],
            2. * X[0] * X[1],
            0. * X[0],
            2. * X[0] * X[2],
            X[1] ** 2,
            0. * X[0],
            X[2] ** 2,
            2. * X[0] * X[1] ** 2,
            0. * X[0],
            2. * X[0] * X[2] ** 2,
            X[1] * X[2],
            2. * X[0] * X[1] * X[2],
            X[1] ** 2 * X[2],
            X[1] * X[2] ** 2,
            2. * X[0] * X[1] ** 2 * X[2],
            X[1] ** 2 * X[2] ** 2,
            2. * X[0] * X[1] * X[2] ** 2,
            2. * X[0] * X[1] ** 2 * X[2] ** 2,
        ])

    def _power_basis_dy(self, X):
        return np.array([
            0. * X[0],
            0. * X[0],
            1. + 0. * X[0],
            0. * X[0],
            0. * X[0],
            2. * X[1],
            0. * X[0],
            X[0],
            X[2],
            0. * X[0],
            X[0] ** 2,
            2. * X[1] * X[2],
            0. * X[0],
            X[0] * 2. * X[1],
            X[2] ** 2,
            0. * X[0],
            X[0] ** 2 * 2. * X[1],
            2. * X[1] * X[2] ** 2,
            0. * X[0],
            X[0] * X[2],
            X[0] ** 2 * X[2],
            X[0] * 2. * X[1] * X[2],
            X[0] * X[2] ** 2,
            X[0] ** 2 * 2. * X[1] * X[2],
            X[0] * 2. * X[1] * X[2] ** 2,
            X[0] ** 2 * X[2] ** 2,
            X[0] ** 2 * 2. * X[1] * X[2] ** 2,
        ])

    def _power_basis_dz(self, X):
        return np.array([
            0. * X[0],
            0. * X[0],
            0. * X[0],
            1. + 0. * X[0],
            0. * X[0],
            0. * X[0],
            2. * X[2],
            0. * X[0],
            X[1],
            X[0],
            0. * X[0],
            X[1] ** 2,
            X[0] ** 2,
            0. * X[0],
            X[1] * 2. * X[2],
            X[0] * 2. * X[2],
            0. * X[0],
            X[1] ** 2 * 2. * X[2],
            X[0] ** 2 * 2. * X[2],
            X[0] * X[1],
            X[0] ** 2 * X[1],
            X[0] * X[1] ** 2,
            X[0] * X[1] * 2. * X[2],
            X[0] ** 2 * X[1] ** 2,
            X[0] * X[1] ** 2 * 2. * X[2],
            X[0] ** 2 * X[1] * 2. * X[2],
            X[0] ** 2 * X[1] ** 2 * 2. * X[2],
        ])

    def lbasis(self, X, i):
        if i >= 27 or i < 0:
            self._index_error()
        return (
            np.einsum('i...,i...', self.invV[i], self._power_basis(X)),
            np.array([
                np.einsum('i...,i...', self.invV[i], self._power_basis_dx(X)),
                np.einsum('i...,i...', self.invV[i], self._power_basis_dy(X)),
                np.einsum('i...,i...', self.invV[i], self._power_basis_dz(X)),
            ])
        )
