import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTri


class ElementTriCCR(ElementH1):
    """The conforming Crouzeix-Raviart element."""

    nodal_dofs = 1
    facet_dofs = 1
    interior_dofs = 1
    maxdeg = 3
    dofnames = ['u', 'u', 'u']
    doflocs = np.array([[0., 0.],
                        [1., 0.],
                        [0., 1.],
                        [.5, 0.],
                        [.5, .5],
                        [0., .5],
                        [1./3, 1./3]])

    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:  # (0., 0.)
            phi = -(-2*x + y*(3*x - 2) + 1)*(x + y - 1)
            dphi = np.array([-6*x*y + 4*x - 3*y**2 + 7*y - 3,
                             -3*x**2 - 6*x*y + 7*x + 4*y - 3])
        elif i == 1:  # (1., 0.)
            phi = -x*(-2*x + 3*y*(x + y - 1) + 1)
            dphi = np.array([
                -6*x*y + 4*x - 3*y**2 + 3*y - 1,
                3*x*(-x - 2*y + 1)
            ])
        elif i == 2:  # (0., 1.)
            phi = -y*(3*x*(x + y - 1) - 2*y + 1)
            dphi = np.array([
                3*y*(-2*x - y + 1),
                -3*x**2 - 6*x*y + 3*x + 4*y - 1
            ])
        elif i == 3:  # 0->2
            phi = 4*x*(3*y - 1)*(x + y - 1)
            dphi = np.array([
                4*(3*y - 1)*(2*x + y - 1),
                4*x*(3*x + 6*y - 4)
            ])
        elif i == 4:  # 0->1
            phi = 4*x*y*(3*x + 3*y - 2)
            dphi = np.array([
                4*y*(6*x + 3*y - 2),
                4*x*(3*x + 6*y - 2)
            ])
        elif i == 5:  # 1->2
            phi = 4*y*(3*x - 1)*(x + y - 1)
            dphi = np.array([
                4*y*(6*x + 3*y - 4),
                4*(3*x - 1)*(x + 2*y - 1)
            ])
        elif i == 6:  # centroid
            phi = 27*x*y*(-x - y + 1)
            dphi = np.array([
                27*y*(-2*x - y + 1),
                27*x*(-x - 2*y + 1)
            ])
        else:
            self._index_error()

        return phi, dphi
