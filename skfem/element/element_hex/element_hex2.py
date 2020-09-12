import numpy as np
from numpy.linalg import inv

from ..element_h1 import ElementH1
from ...mesh.mesh3d import MeshHex


class ElementHex2(ElementH1):

    nodal_dofs = 1
    facet_dofs = 1
    edge_dofs = 1
    interior_dofs = 1
    dim = 3
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
                        [1., .5, .5],
                        [.5, .5, 1.],
                        [.5, 1., .5],
                        [.5, .0, .5],
                        [.5, .5, 0.],
                        [0., .5, .5],
                        [1., 1., .5],
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
                        [.5, .5, .5]])
    mesh_type = MeshHex

    def __init__(self):
        X = self.doflocs.T
        V = self._power_basis(X)
        self.invV = inv(V).T
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
            X[0] * 2. *  X[1] * X[2],
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
        return (
            np.sum(self.invV[i][:, None] * self._power_basis(X), axis=0),
            np.array([
                np.sum(self.invV[i][:, None] * self._power_basis_dx(X), axis=0),
                np.sum(self.invV[i][:, None] * self._power_basis_dy(X), axis=0),
                np.sum(self.invV[i][:, None] * self._power_basis_dz(X), axis=0),
            ])
        )
