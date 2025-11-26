import numpy as np

from ..element_hcurl import ElementHcurl
from ...refdom import RefTri


s_1 = 0.
s_2 = 1.


class ElementTriN2(ElementHcurl):
    """The second order Nedelec element."""

    facet_dofs = 2
    interior_dofs = 2
    maxdeg = 2
    dofnames = ['u^t', 'u^t', 'u^x', 'u^y']
    doflocs = np.array([[s_1, .0],
                        [s_2, .0],
                        [s_2, 1. - s_2],
                        [s_1, 1. - s_1],
                        [.0, s_1],
                        [.0, s_2],
                        [1/3, 1/3],
                        [1/3, 1/3]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = np.array([8.*x*y-6*x+8*y**2-12*y+4,
                            2*x*(-4*x-4*y+3)])
            dphi = -24*y-24*x+18
        elif i == 1:
            phi = np.array([-8*x*y+6*x+2*y-2,
                            4*x*(2*x-1)])
            dphi = 24*x-6
        elif i == 2:
            phi = np.array([2*y*(1-4*x),
                            4*x*(2*x-1)])
            dphi = 24*x-6
        elif i == 3:
            phi = np.array([4*y*(1-2*y),
                            2*x*(4*y-1)])
            dphi = 24*y-6
        elif i == 4:
            phi = np.array([2*y*(-4*x-4*y+3),
                            8*x**2+8*x*y-12*x-6*y+4])
            dphi = 24*y+24*x-18
        elif i == 5:
            phi = np.array([4*y*(2*y-1),
                            -8*x*y+2*x+6*y-2])
            dphi = 6-24*y
        elif i == 6:
            phi = np.array([8*y*(-x-2*y+2),
                            8*x*(x+2*y-1)])
            dphi = 48*y+24*x-24
        elif i == 7:
            phi = np.array([8*y*(2*x+y-1),
                            8*x*(-2*x-y+2)])
            dphi = -24*y-48*x+24
        else:
            self._index_error()

        return phi, dphi
