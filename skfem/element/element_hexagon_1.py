import numpy as np

from .element_h1 import ElementH1
from ..refdom import RefHexagon


class ElementHexagon1(ElementH1):
    """Hexagonal element."""

    nodal_dofs = 1
    maxdeg = 2
    dofnames = ['u']
    doflocs = np.array([[1.0, 0.0],
                        [0.5, np.sqrt(3) / 2],
                        [-0.5, np.sqrt(3) / 2],
                        [-1.0, 0.0],
                        [-0.5, -np.sqrt(3) / 2],
                        [0.5, -np.sqrt(3) / 2]])
    refdom = RefHexagon

    def lbasis(self, X, i):
        x, y = X
        v = 2 * (3 - x ** 2 - y ** 2)

        if i == 0:
            phi = 1 / v * ((1 - 2 * y / np.sqrt(3))
                           * (1 + x - y / np.sqrt(3))
                           * (1 + x + y / np.sqrt(3))
                           * (1 + 2 * y / np.sqrt(3)))
            dphi = np.array([((-3 + 4*y**2)*(-3*(1 + x)*(3 + x) + (3 + 4*x)*y**2))/(9.*(-3 + x**2 + y**2)**2),
                             (2*y*(-2 + (x*(-3 + 2*x)*(3 + 2*x)**2)/(-3 + x**2 + y**2)**2))/9.])
        elif i == 1:
            phi = 1 / v * ((1 - x + y / np.sqrt(3))
                           * (1 + x - y / np.sqrt(3))
                           * (1 + x + y / np.sqrt(3))
                           * (1 + 2 * y / np.sqrt(3)))
            dphi = np.array([-0.018518518518518517*((3 + 2*np.sqrt(3)*y)*(-9*x**4 + 4*x*(-3 + 2*np.sqrt(3)*y)*(-3 + y**2) + 3*(-3 + y**2)*(3 + 2*np.sqrt(3)*y + y**2) - 6*x**2*(-12 + y*(np.sqrt(3) + 5*y))))/(-3 + x**2 + y**2)**2,
                             (np.sqrt(3)*(3 - 2*x) + 4*y + (2*np.sqrt(3)*x*(9 - 4*x**2))/(-3 + x**2 + y**2) + (4*x*(-9 + 4*x**2)*(np.sqrt(3)*x**2 - x*y - 3*(np.sqrt(3) + y)))/(-3 + x**2 + y**2)**2)/18.])
        elif i == 2:
            phi = 1 / v * ((1 + 2 * y / np.sqrt(3))
                           * (1 - x + y / np.sqrt(3))
                           * (1 + x + y / np.sqrt(3))
                           * (1 - x - y / np.sqrt(3)))
            dphi = np.array([-0.018518518518518517*((3 + 2*np.sqrt(3)*y)*(9*x**4 + 4*x*(-3 + 2*np.sqrt(3)*y)*(-3 + y**2) -  3*(-3 + y**2)*(3 + 2*np.sqrt(3)*y + y**2) + 6*x**2*(-12 + y*(np.sqrt(3) + 5*y))))/(-3 + x**2 + y**2)**2, (np.sqrt(3)*(3 + 2*x) + 4*y - (4*x*(-9 + 4*x**2)*(np.sqrt(3)*(-3 + x**2) + (-3 + x)*y))/(-3 + x**2 + y**2)**2 + (2*np.sqrt(3)*x*(-9 + 4*x**2))/(-3 + x**2 + y**2))/18.])
        elif i == 3:
            phi = 1 / v * ((1 - 2 * y / np.sqrt(3))
                           * (1 - x + y / np.sqrt(3))
                           * (1 - x - y / np.sqrt(3))
                           * (1 + 2 * y / np.sqrt(3)))
            dphi = np.array([((-3 + 4*y**2)*(3*(-3 + x)*(-1 + x) + (-3 + 4*x)*y**2))/(9.*(-3 + x**2 + y**2)**2),
                             (2*y*(-2 + ((3 - 2*x)**2*x*(3 + 2*x))/(-3 + x**2 + y**2)**2))/9.])
        elif i == 4:
            phi = 1 / v * ((1 - 2 * y / np.sqrt(3))
                           * (1 + x - y / np.sqrt(3))
                           * (1 - x + y / np.sqrt(3))
                           * (1 - x - y / np.sqrt(3)))
            dphi = np.array([((-3 + 2*np.sqrt(3)*y)*(9*x**4 + 4*x*(-3 + 2*np.sqrt(3)*y)*(-3 + y**2) - 3*(-3 + y**2)*(3 + 2*np.sqrt(3)*y + y**2) + 6*x**2*(-12 + y*(np.sqrt(3) + 5*y))))/(54.*(-3 + x**2 + y**2)**2),
                             (6*np.sqrt(3)*x**5 + x**4*(-9*np.sqrt(3) + 12*y) - (np.sqrt(3) + 4*y)*(-3 + y**2)**2 - 6*np.sqrt(3)*x**3*(3 + 2*y**2) + x**2*(30*np.sqrt(3) - 36*y + 6*np.sqrt(3)*y**2 - 8*y**3) + 2*x*y*(18 - np.sqrt(3)*y*(-9 + y**2)))/(18.*(-3 + x**2 + y**2)**2)])
        elif i == 5:
            phi = 1 / v * ((1 - 2 * y / np.sqrt(3))
                           * (1 + x - y / np.sqrt(3))
                           * (1 + x + y / np.sqrt(3))
                           * (1 - x - y / np.sqrt(3)))
            dphi = np.array([-0.018518518518518517*((-3 + 2*np.sqrt(3)*y)*(9*x**4 - 6*x**2*(12 + (np.sqrt(3) - 5*y)*y) + 4*x*(3 + 2*np.sqrt(3)*y)*(-3 + y**2) + 3*(-3 + 2*np.sqrt(3)*y - y**2)*(-3 + y**2)))/(-3 + x**2 + y**2)**2,
                             (np.sqrt(3)*(-3 + 2*x) + 4*y - (4*x*(-9 + 4*x**2)*(np.sqrt(3)*(-3 + x**2) + (3 + x)*y))/(-3 + x**2 + y**2)**2 + (2*np.sqrt(3)*x*(-9 + 4*x**2))/(-3 + x**2 + y**2))/18.])
        else:
            self._index_error()

        return phi, dphi
