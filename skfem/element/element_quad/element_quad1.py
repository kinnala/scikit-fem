import numpy as np
from ..element_h1 import ElementH1


class ElementQuad1(ElementH1):
    nodal_dofs = 1
    dim = 2
    maxdeg = 2
    dofnames = ['u']

    def lbasis(self, X, i):
        x, y = X[0, :], X[1, :]

        if i == 0:
            phi = 0.25 * (1 - x) * (1 - y)
            dphi = np.array([0.25 * (-1 + y),
                             0.25 * (-1 + x)])
        elif i == 1:
            phi = 0.25 * (1 + x) * (1 - y)
            dphi = np.array([0.25 * (1 - y),
                             0.25 * (-1 - x)])
        elif i == 2:
            phi = 0.25 * (1 + x) * (1 + y)
            dphi = np.array([0.25 * (1 + y),
                             0.25 * (1 + x)])
        elif i == 3:
            phi = 0.25 * (1 - x) * (1 + y)
            dphi = np.array([0.25 * (-1 - y),
                             0.25 * (1 - x)])
        else:
            raise Exception("!")

        return phi, dphi
