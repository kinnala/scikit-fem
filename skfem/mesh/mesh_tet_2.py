from dataclasses import dataclass, replace
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementTetP2
from .mesh_tet_1 import MeshTet1


@dataclass(repr=False)
class MeshTet2(MeshTet1):
    """A quadratic tetrahedral mesh."""

    doflocs: ndarray = np.array([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 1., 1.],
                                 [1., 0., 1.],
                                 [1., 1., 0.],
                                 [1., 1., 1.],
                                 [0., 0., .5],
                                 [0., .5, 0.],
                                 [.5, 0., 0.],
                                 [0., .5, .5],
                                 [.5, 0., .5],
                                 [0., .5, 1.],
                                 [.5, 0., 1.],
                                 [.5, .5, 1.],
                                 [.5, .5, 0.],
                                 [0., 1., .5],
                                 [.5, 1., 0.],
                                 [.5, 1., .5],
                                 [1., 0., .5],
                                 [1., .5, 0.],
                                 [1., .5, .5],
                                 [.5, 1., 1.],
                                 [1., .5, 1.],
                                 [1., 1., .5]], dtype=np.float64).T
    elem: Type[Element] = ElementTetP2

    @classmethod
    def init_ball(cls: Type, nrefs: int = 3) -> 'MeshTet2':
        m = MeshTet1.init_ball(nrefs=nrefs)
        M = cls.from_mesh(m)
        D = M.dofs.get_facet_dofs(M.boundary_facets()).flatten()
        doflocs = M.doflocs.copy()
        doflocs[:, D] /= np.linalg.norm(doflocs[:, D], axis=0)
        return replace(M, doflocs=doflocs)
