import numpy as np
from ..element_h1 import ElementH1


class ElementHex1(ElementH1):
    nodal_dofs = 1
    dim = 3
    maxdeg = 3
    dofnames = ['u']

    def lbasis(self, X, i):
        x, y, z = X[0, :], X[1, :], X[2, :]

        if i == 0:
            phi = 0.125*(1 - x)*(1 - y)*(1 - z)
            dphi = np.array([0.125*(-1 + y)*(1 - z),
                             0.125*(-1 + x)*(1 - z),
                             -0.125*(1 - x)*(1 - y)])
        elif i == 1:
            phi = 0.125*(1 + x)*(1 - y)*(1 - z)
            dphi = np.array([0.125*(1 - y)*(1 - z),
                             0.125*(-1 - x)*(1 - z),
                             -0.125*(1 + x)*(1 - y)])
        elif i == 2:
            phi = 0.125*(1 + x)*(1 + y)*(1 - z)
            dphi = np.array([0.125*(1 + y)*(1 - z),
                             0.125*(1 + x)*(1 - z),
                             -0.125*(1 + x)*(1 + y)])
        elif i == 3:
            phi = 0.125*(1 - x)*(1 + y)*(1 - z)
            dphi = np.array([0.125*(-1 - y)*(1 - z),
                             0.125*(1 - x)*(1 - z),
                             -0.125*(1 - x)*(1 + y)])
        elif i == 4:
            phi = 0.125*(1 - x)*(1 - y)*(1 + z)
            dphi = np.array([0.125*(-1 + y)*(1 + z),
                             0.125*(-1 + x)*(1 + z),
                             0.125*(1 - x)*(1 - y)])
        elif i == 5:
            phi = 0.125*(1 + x)*(1 - y)*(1 + z)
            dphi = np.array([0.125*(1 - y)*(1 + z),
                             0.125*(-1 - x)*(1 + z),
                             0.125*(1 + x)*(1 - y)])
        elif i == 6:
            phi = 0.125*(1 + x)*(1 + y)*(1 + z)
            dphi = np.array([0.125*(1 + y)*(1 + z),
                             0.125*(1 + x)*(1 + z),
                             0.125*(1 + x)*(1 + y)])
        elif i == 7:
            phi = 0.125*(1 - x)*(1 + y)*(1 + z)
            dphi = np.array([0.125*(-1 - y)*(1 + z),
                             0.125*(1 - x)*(1 + z),
                             0.125*(1 - x)*(1 + y)])
        else:
            raise Exception("!")

        return phi, dphi
