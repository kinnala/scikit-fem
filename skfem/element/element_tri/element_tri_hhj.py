import numpy as np

from ..element_matrix import ElementMatrix
from ...refdom import RefTri


class ElementTriHHJ(ElementMatrix):
    """The first order Herman-Hellan-Johnson element."""

    facet_dofs = 2
    interior_dofs = 3
    maxdeg = 1
    dofnames = ['u^n^2', 'u^n^1', 'u^i^1', 'u^i^2', 'u^i^3']
    doflocs = np.array([[.5, .0],
                        [.5, .0],
                        [.5, .5],
                        [.5, .5],
                        [.0, .5],
                        [.0, .5],
                        [.33, .33],
                        [.33, .33],
                        [.33, .33]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = np.array([[0*x, 3*x+3*y-2],
                            [3*x+3*y-2, -6*x-6*y+4]])
        elif i == 1:
            phi = np.array([[0*x, 1 - 3*x],
                            [1 - 3*x, 6*x - 2]])
        elif i == 2:
            phi = np.array([[0*x, 3*x-1],
                            [3*x-1, 0*x]])
        elif i == 3:
            phi = np.array([[0*x, 3*y-1],
                            [3*y-1, 0*x]])
        elif i == 4:
            phi = np.array([[-6*x-6*y+4, 3*x+3*y-2],
                            [3*x+3*y-2,0*x]])
        elif i == 5:
            phi = np.array([[6*y-2,1-3*y],
                            [1-3*y,0*x]])
        # interior
        elif i == 6:
            phi = np.array([[6*x, -6*x-3*y+3],
                            [-6*x-3*y+3, 0*x]])
        elif i == 7:
            phi = np.array([[0*x, -3*x-3*y+3],
                            [-3*x-3*y+3, 0*x]])
        elif i == 8:
            phi = np.array([[0*x, -3*x-6*y+3],
                            [-3*x-6*y+3, 6*y]])
        else:
            self._index_error()

        return phi, None
