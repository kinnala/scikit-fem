import numpy as np
from ..element_h1 import ElementH1


class ElementLineP1(ElementH1):
    nodal_dofs = 1
    dim = 1
    maxdeg = 1
    dofnames = ['u']

    def lbasis(self, X, i):
        x = X[0, :]

        if i == 0:
            phi = 1 - x
            dphi = np.array([-1 + 0*x])
        elif i == 1:
            phi = x
            dphi = np.array([1 + 0*x])
        else:
            raise Exception("!")

        return phi, dphi
