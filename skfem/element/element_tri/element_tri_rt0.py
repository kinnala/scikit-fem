import numpy as np
from ..element_hdiv import ElementHdiv


class ElementTriRT0(ElementHdiv):
    facet_dofs = 1
    dim = 2
    maxdeg = 1
    dofnames = ['u^n']

    def lbasis(self, X, i):
        x, y = X[0, :], X[1, :]

        if i == 0:
            phi = np.array([x, y - 1])
            dphi = 2 + 0*x
        elif i == 1:
            phi = np.array([x, y])
            dphi = 2 + 0*x
        elif i == 2:
            phi = np.array([x - 1, y])
            dphi = 2 + 0*x
        else:
            raise Exception("!")

        return phi, dphi
