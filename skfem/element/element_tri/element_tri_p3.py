import numpy as np

from ..element_h1 import ElementH1
from ...refdom import RefTri


class ElementTriP3(ElementH1):
    """Piecewise cubic element."""

    nodal_dofs = 1
    facet_dofs = 2
    interior_dofs = 1
    maxdeg = 3
    dofnames = ["u", "u", "u", "u"]
    doflocs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / 3, 0.0],
            [2.0 / 3, 0.0],
            [2.0 / 3, 1.0 / 3],
            [1.0 / 3, 2.0 / 3],
            [0.0, 1.0 / 3],
            [0.0, 2.0 / 3],
            [1.0 / 3, 1.0 / 3],
        ]
    )

    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:  # (0., 0.)
            phi = (
                1 / 2 * (1 - x - y) * (2 - 3 * x - 3 * y) * (1 - 3 * x - 3 * y)
            )
            dphi = np.array(
                [
                    18 * x
                    + 18 * y
                    - 27 * x * y
                    - (27 * x ** 2) / 2
                    - (27 * y ** 2) / 2
                    - 11 / 2,
                    18 * x
                    + 18 * y
                    - 27 * x * y
                    - (27 * x ** 2) / 2
                    - (27 * y ** 2) / 2
                    - 11 / 2,
                ]
            )
        elif i == 1:  # (1., 0.)
            phi = 1.0 / 2 * x * (3 * x - 1) * (3 * x - 2)
            dphi = np.array([(27 * x ** 2) / 2 - 9 * x + 1, 0.0 * x])
        elif i == 2:  # (0., 1.)
            phi = 1.0 / 2 * y * (3 * y - 1) * (3 * y - 2)
            dphi = np.array([0.0 * x, (27 * y ** 2) / 2 - 9 * y + 1])
        elif i == 3:  # 0->1: (1/3,0)
            phi = 9.0 / 2 * x * (1 - x - y) * (2 - 3 * x - 3 * y)
            dphi = np.array(
                [
                    (81 * x ** 2) / 2
                    + 54 * x * y
                    - 45 * x
                    + (27 * y ** 2) / 2
                    - (45 * y) / 2
                    + 9,
                    (9 * x * (6 * x + 6 * y - 5)) / 2,
                ]
            )
        elif i == 5:  # 1->2: (2/3,1/3)
            phi = 9.0 / 2 * x * y * (3 * x - 1)
            dphi = np.array(
                [(9 * y * (6 * x - 1)) / 2, (9 * x * (3 * x - 1)) / 2]
            )
        elif i == 8:  # 0->2: (0,2/3)
            phi = 9.0 / 2 * y * (1 - x - y) * (3 * y - 1)
            dphi = np.array(
                [
                    -(9 * y * (3 * y - 1)) / 2,
                    (9 * x) / 2
                    + 36 * y
                    - 27 * x * y
                    - (81 * y ** 2) / 2
                    - 9 / 2,
                ]
            )
        elif i == 4:  # 0->1: (2/3,0)
            phi = 9.0 / 2 * x * (1 - x - y) * (3 * x - 1)
            dphi = np.array(
                [
                    36 * x
                    + (9 * y) / 2
                    - 27 * x * y
                    - (81 * x ** 2) / 2
                    - 9 / 2,
                    -(9 * x * (3 * x - 1)) / 2,
                ]
            )
        elif i == 6:  # 1->2: (1/3,2/3)
            phi = 9.0 / 2 * x * y * (3 * y - 1)
            dphi = np.array(
                [(9 * y * (3 * y - 1)) / 2, (9 * x * (6 * y - 1)) / 2]
            )
        elif i == 7:  # 0->2: (0,1/3)
            phi = 9 / 2 * y * (1 - x - y) * (2 - 3 * x - 3 * y)
            dphi = np.array(
                [
                    (9 * y * (6 * x + 6 * y - 5)) / 2,
                    (27 * x ** 2) / 2
                    + 54 * x * y
                    - (45 * x) / 2
                    + (81 * y ** 2) / 2
                    - 45 * y
                    + 9,
                ]
            )
        elif i == 9:  # centroid (1/3,1/3)
            phi = 27 * x * y * (-x - y + 1)
            dphi = np.array(
                [27 * y * (-2 * x - y + 1), 27 * x * (-x - 2 * y + 1)]
            )
        else:
            self._index_error()

        return phi, dphi
