import numpy as np
from ..element_hdiv import ElementHdiv


class ElementTetRT0(ElementHdiv):
    facet_dofs = 1
    dim = 3
    maxdeg = 1
    dofnames = ['u^n']

    def lbasis(self, X, i):
        x, y, z = X[0, :], X[1, :], X[2, :]

        if i == 0:
            phi = np.array([x, y, z-1])
            dphi = 3 + 0*x
        elif i == 1:
            phi = np.array([x, y-1, z])
            dphi = 3 + 0*x
        elif i == 2:
            phi = np.array([x-1, y, z])
            dphi = 3 + 0*x
        elif i == 3:
            phi = np.array([x, y, z])
            dphi = 3 + 0*x
        else:
            raise Exception("!")

        return phi, dphi
