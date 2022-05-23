import numpy as np

from ..element_hdiv import ElementHdiv
from ...refdom import RefTri


class ElementTriRT2(ElementHdiv):
    """The quadratic Raviart-Thomas element."""

    facet_dofs = 2
    interior_dofs = 2
    maxdeg = 2
    dofnames = ['u^n', 'u^n', 'NA', 'NA']
    doflocs = np.array([[0., 0.],
                        [1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 0.],
                        [0., 1.],
                        [1/3, 1/3],
                        [1/3, 1/3]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:  # bottom
            phi = -np.array([2*x*(4*x+4*y-3), 8*x*y-6*x+8*y**2-12*y+4])
            dphi = -6 * (-3 + 4*x + 4*y)
        elif i == 1:
            phi = -np.array([4*x*(1-2*x), -8*x*y+6*x+2*y-2])
            dphi = -(6 - 24*x)
        elif i == 2:  # top-right
            phi = -np.array([4*x*(1-2*x), 2*y*(1-4*x)])
            dphi = -(6 - 24*x)
        elif i == 3:
            phi = -np.array([2*x*(1-4*y), 4*y*(1-2*y)])
            dphi = -(6 - 24*y)
        elif i == 4:  # left
            phi = np.array([-8*x**2-8*x*y+12*x+6*y-4, 2*y*(-4*x-4*y+3)])
            dphi = -6 * (-3 + 4*x + 4*y)
        elif i == 5:
            phi = np.array([8*x*y-2*x-6*y+2, 4*y*(2*y-1)])
            dphi = -6 + 24*y
        elif i == 6:  # interior u_x
            phi = np.array([8*x*(-2*x-y+2), 8*y*(-2*x-y+1)])
            dphi = -24*(-1 + 2*x + y)
        elif i == 7:  # interior u_y
            phi = np.array([8*x*(-x-2*y+1), 8*y*(-x-2*y+2)])
            dphi = -24*(-1 + x + 2*y)
        else:
            self._index_error()

        return phi, dphi
