import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTet


class ElementTetCCR(ElementH1):
    """Conforming Crouzeix-Raviart element."""

    nodal_dofs = 1
    facet_dofs = 1
    edge_dofs = 1
    interior_dofs = 1
    maxdeg = 4
    dofnames = ["u", "u", "u", "u"]

    doflocs = np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [.5, 0., 0.],
                        [.5, .5, 0.],
                        [0., .5, 0.],
                        [0., 0., .5],
                        [.5, 0., .5],
                        [0., .5, .5],
                        [1./3, 1./3, 0.],
                        [1./3, 0., 1./3],
                        [0., 1./3, 1. / 3],
                        [1./3, 1./3., 1./3],
                        [0.25, 0.25, 0.25]])
    refdom = RefTet

    def lbasis(self, X, i):
        x, y, z = X

        if i == 0:  # at (0,0,0)
            phi = (x + y + z - 1)*(4*x*y*z - 3*x*y - 3*x*z + 2*x - 3*y*z +
                                   2*y + 2*z - 1)
            dphi = np.array([
                8*x*y*z - 6*x*y - 6*x*z + 4*x + 4*y**2*z - 3*y**2 + 4*y*z**2 -
                13*y*z + 7*y - 3*z**2 + 7*z - 3,
                4*x**2*z - 3*x**2 + 8*x*y*z - 6*x*y + 4*x*z**2 - 13*x*z + 7*x -
                6*y*z + 4*y - 3*z**2 + 7*z - 3,
                4*x**2*y - 3*x**2 + 4*x*y**2 + 8*x*y*z - 13*x*y - 6*x*z + 7*x -
                3*y**2 - 6*y*z + 7*y + 4*z - 3
            ])
        elif i == 1:  # at (1,0,0)
            phi = 3 * x * (y * z + (y + z) * (-x - y - z + 1)) + x * (
                  2 * x - 4 * y * z * (-x - y - z + 1) - 1)
            dphi = np.array([
                -3*x*(y + z) + 2*x*(2*y*z + 1) + 2*x + 4*y*z*(x + y + z - 1) +
                3*y*z - 3*(y + z)*(x + y + z - 1) - 1,
                x*(4*z - 3)*(x + 2*y + z - 1),
                x*(4*y - 3)*(x + y + 2*z - 1)
            ])
        elif i == 2:  # at (0,1,0)
            phi = y*(4*x*z*(x + y + z - 1) + 3*x*z + 2*y -
                     3*(x + z)*(x + y + z - 1) - 1)
            dphi = np.array([
                y*(4*z - 3)*(2*x + y + z - 1),
                4*x*z*(x + y + z - 1) + 3*x*z - 3*y*(x + z) + 2*y*(2*x*z + 1) +
                2*y - 3*(x + z)*(x + y + z - 1) - 1,
                y*(4*x - 3)*(x + y + 2*z - 1)
            ])
        elif i == 3:  # at (0,0,1)
            phi = z*(4*x*y*(x + y + z - 1) + 3*x*y +
                     2*z - 3*(x + y)*(x + y + z - 1) - 1)
            dphi = np.array([
                z*(4*y - 3)*(2*x + y + z - 1),
                z*(4*x - 3)*(x + 2*y + z - 1),
                4*x*y*(x + y + z - 1) + 3*x*y - 3*z*(x + y) + 2*z*(2*x*y + 1) +
                2*z - 3*(x + y)*(x + y + z - 1) - 1
            ])
        elif i == 4:  # between (0,1)
            phi = -4*x*(x + y + z - 1)*(8*y*z - 3*y - 3*z + 1)
            dphi = np.array([
                4*(-2*x - y - z + 1)*(8*y*z - 3*y - 3*z + 1),
                4*x*(-8*y*z + 3*y + 3*z - (8*z - 3)*(x + y + z - 1) - 1),
                4*x*(-8*y*z + 3*y + 3*z - (8*y - 3)*(x + y + z - 1) - 1)
            ])
        elif i == 5:  # between (1,2)
            phi = 4*x*y*(3*x + 3*y - 8*z*(x + y + z - 1) - 2)
            dphi = np.array([
                4*y*(-x*(8*z - 3) + 3*x + 3*y - 8*z*(x + y + z - 1) - 2),
                4*x*(3*x - y*(8*z - 3) + 3*y - 8*z*(x + y + z - 1) - 2),
                32*x*y*(-x - y - 2*z + 1)
            ])
        elif i == 6:  # between (0,2)
            phi = -4*y*(x + y + z - 1)*(8*x*z - 3*x - 3*z + 1)
            dphi = np.array([
                4*y*(-8*x*z + 3*x + 3*z - (8*z - 3)*(x + y + z - 1) - 1),
                4*(-x - 2*y - z + 1)*(8*x*z - 3*x - 3*z + 1),
                4*y*(-8*x*z + 3*x + 3*z - (8*x - 3)*(x + y + z - 1) - 1)
            ])
        elif i == 7:  # between (0,3)
            phi = -4*z*(x + y + z - 1)*(8*x*y - 3*x - 3*y + 1)
            dphi = np.array([
                4*z*(-8*x*y + 3*x + 3*y - (8*y - 3)*(x + y + z - 1) - 1),
                4*z*(-8*x*y + 3*x + 3*y - (8*x - 3)*(x + y + z - 1) - 1),
                4*(-x - y - 2*z + 1)*(8*x*y - 3*x - 3*y + 1)
            ])
        elif i == 8:
            phi = 4*x*z*(3*x - 8*y*(x + y + z - 1) + 3*z - 2)
            dphi = np.array([
                4*z*(-x*(8*y - 3) + 3*x - 8*y*(x + y + z - 1) + 3*z - 2),
                32*x*z*(-x - 2*y - z + 1),
                4*x*(3*x - 8*y*(x + y + z - 1) - z*(8*y - 3) + 3*z - 2)
            ])
        elif i == 9:
            phi = -4*y*z*(8*x*(x + y + z - 1) - 3*y - 3*z + 2)
            dphi = np.array([
                32*y*z*(-2*x - y - z + 1),
                4*z*(-8*x*(x + y + z - 1) - y*(8*x - 3) + 3*y + 3*z - 2),
                4*y*(-8*x*(x + y + z - 1) + 3*y - z*(8*x - 3) + 3*z - 2)
            ])
        elif i == 10:
            phi = 27*x*y*(4*z - 1)*(x + y + z - 1)
            dphi = np.array([
                27*y*(4*z - 1)*(2*x + y + z - 1),
                27*x*(4*z - 1)*(x + 2*y + z - 1),
                27*x*y*(4*x + 4*y + 8*z - 5)
            ])
        elif i == 11:
            phi = 27*x*z*(4*y - 1)*(x + y + z - 1)
            dphi = np.array([
                27*z*(4*y - 1)*(2*x + y + z - 1),
                27*x*z*(4*x + 8*y + 4*z - 5),
                27*x*(4*y - 1)*(x + y + 2*z - 1)
            ])
        elif i == 12:
            phi = 27*y*z*(4*x - 1)*(x + y + z - 1)
            dphi = np.array([
                27*y*z*(8*x + 4*y + 4*z - 5),
                27*z*(4*x - 1)*(x + 2*y + z - 1),
                27*y*(4*x - 1)*(x + y + 2*z - 1)
            ])
        elif i == 13:
            phi = 27*x*y*z*(4*x + 4*y + 4*z - 3)
            dphi = np.array([
                27*y*z*(8*x + 4*y + 4*z - 3),
                27*x*z*(4*x + 8*y + 4*z - 3),
                27*x*y*(4*x + 4*y + 8*z - 3)
            ])
        elif i == 14:
            phi = 256*x*y*z*(-x - y - z + 1)
            dphi = np.array([
                256*y*z*(-2*x - y - z + 1),
                256*x*z*(-x - 2*y - z + 1),
                256*x*y*(-x - y - 2*z + 1)
            ])
        else:
            self._index_error()

        return phi, dphi
