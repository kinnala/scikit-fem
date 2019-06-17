import numpy as np
from ..element_h1 import ElementH1


class ElementTriP1(ElementH1):
    nodal_dofs = 1
    dim = 2
    maxdeg = 1
    dofnames = ['u']

    def lbasis(self, X, i):
        x, y = X[0, :], X[1, :]

        if i == 0:
            phi = 1 - x - y
            dphi = np.array([-1 + 0*x, -1 + 0*x])
        elif i == 1:
            phi = x
            dphi = np.array([1 + 0*x, 0*x])
        elif i == 2:
            phi = y
            dphi = np.array([0*x, 1 + 0*x])
        else:
            raise Exception("!")

        return phi, dphi
