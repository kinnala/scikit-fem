import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTet


class ElementTetP2(ElementH1):

    nodal_dofs = 1
    edge_dofs = 1
    maxdeg = 2
    dofnames = ['u', 'u']
    doflocs = np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [.5, 0., 0.],
                        [.5, .5, 0.],
                        [0., .5, 0.],
                        [0., .0, .5],
                        [.5, .0, .5],
                        [.0, .5, .5]])
    refdom = RefTet

    def lbasis(self, X, i):
        x, y, z = X

        if i == 0:  # at (0,0,0)
            phi = (1. - 3.*x + 2.*x**2 - 3.*y + 4.*x*y +
                   2.*y**2 - 3.*z + 4.*x*z + 4.*y*z + 2.*z**2)
            dphi = np.array([
                -3. + 4.*x + 4.*y + 4.*z,
                -3. + 4.*x + 4.*y + 4.*z,
                -3. + 4.*x + 4.*y + 4.*z,
            ])
        elif i == 1:  # at (1,0,0)
            phi = - 1.*x + 2.*x**2
            dphi = np.array([
                -1 + 4*x,
                0*x,
                0*x,
            ])
        elif i == 2:  # at (0,1,0)
            phi = - 1.*y + 2.*y**2
            dphi = np.array([
                0*x,
                -1. + 4.*y,
                0*x,
            ])
        elif i == 3:  # at (0,0,1)
            phi = - 1.*z + 2.*z**2
            dphi = np.array([
                0*x,
                0*x,
                -1. + 4.*z,
            ])
        elif i == 4:  # between (0,1)
            phi = 4.*x - 4.*x**2 - 4.*x*y - 4*x*z
            dphi = np.array([
                4. - 8.*x - 4.*y - 4.*z,
                -4.*x,
                -4.*x,
            ])
        elif i == 5:  # between (1,2)
            phi = 4.*x*y
            dphi = np.array([
                4.*y,
                4.*x,
                0*x,
            ])
        elif i == 6:  # between (0,2)
            phi = 0. + 4.*y - 4.*x*y - 4.*y**2 - 4.*y*z
            dphi = np.array([
                -4.*y,
                4. - 4.*x - 8.*y - 4.*z,
                -4.*y,
            ])
        elif i == 7:  # between (0,3)
            phi = 0. + 4.*z - 4.*x*z - 4.*y*z - 4.*z**2
            dphi = np.array([
                -4.*z,
                -4.*z,
                4. - 4.*x - 4.*y - 8.*z,
            ])
        elif i == 8:
            phi = 0. + 4.*x*z
            dphi = np.array([
                4.*z,
                0*x,
                4*x,
            ])
        elif i == 9:
            phi = 0. + 4.*y*z
            dphi = np.array([
                0*x,
                4*z,
                4*y,
            ])
        else:
            self._index_error()

        return phi, dphi
