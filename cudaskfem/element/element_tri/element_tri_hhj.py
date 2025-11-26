import numpy as np

from ..element_matrix import ElementMatrix
from ...refdom import RefTri


class ElementTriHHJ1(ElementMatrix):
    """Piecewise linear Hellan-Herrmann-Johnson element."""

    facet_dofs = 2
    interior_dofs = 3
    maxdeg = 1
    dofnames = ['u^n', 'u^n', 'NA', 'NA', 'NA']
    doflocs = np.array([[1/3, .0],
                        [2/3, .0],
                        [2/3, 1/3],
                        [1/3, 2/3],
                        [.0, 1/3],
                        [.0, 2/3],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [np.nan, np.nan]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = np.array([[0*x, 3*x+3*y-2],
                            [3*x+3*y-2, -6*x-6*y+4]])
        elif i == 1:
            phi = np.array([[0*x, 1-3*x],
                            [1-3*x, 6*x-2]])
        elif i == 2:
            phi = np.array([[0*x, 3*x-1],
                            [3*x-1, 0*x]])
        elif i == 3:
            phi = np.array([[0*x, 3*y-1],
                            [3*y-1, 0*x]])
        elif i == 4:
            phi = np.array([[-6*x-6*y+4, 3*x+3*y-2],
                            [3*x+3*y-2, 0*x]])
        elif i == 5:
            phi = np.array([[6*y-2, 1-3*y],
                            [1-3*y, 0*x]])
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


class ElementTriHHJ0(ElementMatrix):
    """Piecewise constant Hellan-Herrmann-Johnson element."""

    facet_dofs = 1
    maxdeg = 1
    dofnames = ['u^n']
    doflocs = np.array([[0.5, 0.0],
                        [0.5, 0.5],
                        [0.0, 0.5]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = np.array([[0*x, -.5 + 0*x],
                            [-.5 + 0*x, 1 + 0*x]])
        elif i == 1:
            phi = np.array([[0*x, .5 + 0*x],
                            [.5 + 0*x, 0*x]])
        elif i == 2:
            phi = np.array([[1 + 0*x, -.5 + 0*x],
                            [-.5 + 0*x, 0*x]])
        else:
            self._index_error()

        return phi, None
