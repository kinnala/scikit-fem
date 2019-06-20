import numpy as np
from ..element_h1 import ElementH1


class ElementQuad2(ElementH1):
    nodal_dofs = 1
    facet_dofs = 1
    interior_dofs = 1
    dim = 2
    maxdeg = 3
    dofnames = ['u', 'u', 'u']

    def lbasis(self, X, i):
        x, y = X[0, :], X[1, :]

        if i == 0:
            phi = 0.25*(x**2-x)*(y**2-y)
            dphi = np.array([((-1 + 2*x)*(-1 + y)*y)/4.,
                             ((-1 + x)*x*(-1 + 2*y))/4.])
        elif i == 1:
            phi = 0.25*(x**2+x)*(y**2-y)
            dphi = np.array([((1 + 2*x)*(-1 + y)*y)/4.,
                             (x*(1 + x)*(-1 + 2*y))/4. ])
        elif i == 2:
            phi = 0.25*(x**2+x)*(y**2+y)
            dphi = np.array([((1 + 2*x)*y*(1 + y))/4.,
                             (x*(1 + x)*(1 + 2*y))/4.])
        elif i == 3:
            phi = 0.25*(x**2-x)*(y**2+y)
            dphi = np.array([((-1 + 2*x)*y*(1 + y))/4.,
                             ((-1 + x)*x*(1 + 2*y))/4.])
        elif i == 4:
            phi = 0.5*(y**2-y)*(1-x**2)
            dphi = np.array([-(x*(-1 + y)*y),
                             -((-1 + x**2)*(-1 + 2*y))/2.])
        elif i == 5:
            phi = 0.5*(x**2+x)*(1-y**2)
            dphi = np.array([-((1 + 2*x)*(-1 + y**2))/2.,
                             -(x*(1 + x)*y)])
        elif i == 6:
            phi = 0.5*(y**2+y)*(1-x**2)
            dphi = np.array([-(x*y*(1 + y)),
                             -((-1 + x**2)*(1 + 2*y))/2.])
        elif i == 7:
            phi = 0.5*(x**2-x)*(1-y**2)
            dphi = np.array([-((-1 + 2*x)*(-1 + y**2))/2.,
                             -((-1 + x)*x*y)])
        elif i == 8:
            phi = (1-x**2)*(1-y**2)
            dphi = np.array([2*x*(-1 + y**2),
                             2*(-1 + x**2)*y])
        else:
            raise Exception("!")

        return phi, dphi
