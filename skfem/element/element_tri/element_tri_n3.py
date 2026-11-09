import numpy as np

from ..element_hcurl import ElementHcurl
from ...refdom import RefTri


class ElementTriN3(ElementHcurl):
    """The third order Nedelec element."""

    facet_dofs = 3
    interior_dofs = 6
    maxdeg = 3
    dofnames = ['u^t'] * 3 + ['u^x', 'u^y'] * 6
    doflocs = np.array([
            [0.25, 0.0],
            [0.50, 0.0],
            [0.75, 0.0],
            [0.75, 0.25],
            [0.50, 0.50],
            [0.25, 0.75],
            [0.0, 0.25],
            [0.0, 0.50],
            [0.0, 0.75],
            [0.25, 0.25],
            [0.25, 0.25],
            [0.50, 0.25],
            [0.50, 0.25],
            [0.25, 0.50],
            [0.25, 0.50]
        ])

    refdom = RefTri

    def reversed_facet_dof(self, j):
        # the facet DOFs are symmetric with respect to the facet midpoint
        return self.facet_dofs - 1 - j, 1.

    def lbasis(self, X, i):
        if 6 <= i <= 8:
            # the facet DOFs of the third facet are defined from the vertex 2
            # towards the vertex 0 in _lbasis; reorder and change the
            # direction so that all facets are traversed from facets[k][0]
            # to facets[k][1]
            phi, dphi = self._lbasis(X, 14 - i)
            return -phi, -dphi
        return self._lbasis(X, i)

    def _lbasis(self, X, i):
        x, y = X

        x2 = x**2
        y2 = y**2
        xy = x * y

        if i == 0:
            phi = np.array([
                -8.0*x2*y + 8.0*x2 - 16.0*x*y2 + 26.0*xy -
                10.0*x - 8.0*y**3 + 18.0*y2 - 13.0*y + 3.0,
                x*(8.0*x2 + 16.0*xy - 10.0*x + 8.0*y2 - 10.0*y + 3.0)
            ])
            dphi = 32.0*x2 + 64.0*xy - 46.0*x + 32.0*y2 - 46.0*y + 16.0

        elif i == 1:
            phi = np.array([
                16.0*x2*y - 16.0*x2 + 16.0*x*y2 - 32.0*xy + 16.0*x
                - 4.0*y2 + 7.0*y - 3.0,
                x*(-16.0*x2 - 16.0*xy + 16.0*x + 4.0*y - 3.0)
            ])
            dphi = -64.0*x2 - 64.0*xy + 64.0*x + 12.0*y - 10.0

        elif i == 2:
            phi = np.array([
                -8.0*x2*y + 8.0*x2 + 6.0*xy - 6.0*x - 1.0*y + 1.0,
                x*(8.0*x2 - 6.0*x + 1.0)
            ])
            dphi = 32.0*x2 - 18.0*x + 2.0

        elif i == 3:
            phi = np.array([
                y*(-8.0*x2 + 6.0*x - 1.0),
                x*(8.0*x2 - 6.0*x + 1.0)
            ])
            dphi = 32.0*x2 - 18.0*x + 2.0

        elif i == 4:
            phi = np.array([
                y*(-16.0*xy + 4.0*x + 4.0*y - 1.0),
                x*(16.0*xy - 4.0*x - 4.0*y + 1.0)
            ])
            dphi = 64.0*xy - 12.0*x - 12.0*y + 2.0

        elif i == 5:
            phi = np.array([
                y*(-8.0*y2 + 6.0*y - 1.0),
                x*(8.0*y2 - 6.0*y + 1.0)
            ])
            dphi = 32.0*y2 - 18.0*y + 2.0

        elif i == 6:
            phi = np.array([
                y*(-8.0*y2 + 6.0*y - 1.0),
                8.0*x*y2 - 6.0*xy + 1.0*x - 8.0*y2 + 6.0*y - 1.0
            ])
            dphi = 32.0*y2 - 18.0*y + 2.0

        elif i == 7:
            phi = np.array([
                y*(16.0*xy - 4.0*x + 16.0*y2 - 16.0*y + 3.0),
                -16.0*x2*y + 4.0*x2 - 16.0*x*y2 + 32.0*xy
                - 7.0*x + 16.0*y2 - 16.0*y + 3.0
            ])
            dphi = -64.0*xy + 12.0*x - 64.0*y2 + 64.0*y - 10.0

        elif i == 8:
            phi = np.array([
                y*(-8.0*x2 - 16.0*xy + 10.0*x - 8.0*y2 + 10.0*y - 3.0),
                8.0*x**3 + 16.0*x2*y - 18.0*x2 + 8.0*x*y2 -
                26.0*xy + 13.0*x - 8.0*y2 + 10.0*y - 3.0
            ])
            dphi = 32.0*x2 + 64.0*xy - 46.0*x + 32.0*y2 - 46.0*y + 16.0

        elif i == 9:
            phi = np.array([
                y*(8.0*x2 + 32.0*xy - 30.0*x + 24.0*y2 - 42.0*y + 18.0),
                x*(-8.0*x2 - 32.0*xy + 14.0*x - 24.0*y2 + 26.0*y - 6.0)
            ])
            dphi = -32.0*x2 - 128.0*xy + 58.0*x - 96.0*y2 + 110.0*y - 24.0

        elif i == 10:
            phi = np.array([
                y*(-24.0*x2 - 32.0*xy + 26.0*x - 8.0*y2 + 14.0*y - 6.0),
                x*(24.0*x2 + 32.0*xy - 42.0*x + 8.0*y2 - 30.0*y + 18.0)
            ])
            dphi = 96.0*x2 + 128.0*xy - 110.0*x + 32.0*y2 - 58.0*y + 24.0

        elif i == 11:
            phi = np.array([
                y*(-16.0*x2 - 32.0*xy + 36.0*x + 8.0*y - 8.0),
                x*(16.0*x2 + 32.0*xy - 20.0*x - 8.0*y + 4.0)
            ])
            dphi = 64.0*x2 + 128.0*xy - 76.0*x - 24.0*y + 12.0

        elif i == 12:
            phi = np.array([
                y*(24.0*x2 + 16.0*xy - 22.0*x - 4.0*y + 4.0),
                x*(-24.0*x2 - 16.0*xy + 30.0*x + 4.0*y - 6.0)
            ])
            dphi = -96.0*x2 - 64.0*xy + 82.0*x + 12.0*y - 10.0

        elif i == 13:
            phi = np.array([
                y*(-16.0*xy + 4.0*x - 24.0*y2 + 30.0*y - 6.0),
                x*(16.0*xy - 4.0*x + 24.0*y2 - 22.0*y + 4.0)
            ])
            dphi = 64.0*xy - 12.0*x + 96.0*y2 - 82.0*y + 10.0

        elif i == 14:
            phi = np.array([
                y*(32.0*xy - 8.0*x + 16.0*y2 - 20.0*y + 4.0),
                x*(-32.0*xy + 8.0*x - 16.0*y2 + 36.0*y - 8.0)
            ])
            dphi = -128.0*xy + 24.0*x - 64.0*y2 + 76.0*y - 12.0

        else:
            self._index_error()

        return phi, dphi
