from dataclasses import dataclass, replace, field
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementTetP2
from .mesh_tet_1 import MeshTet1


@dataclass(repr=False)
class MeshTet2(MeshTet1):
    """A quadratic tetrahedral mesh."""

    doflocs: ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 1.0],
                [0.5, 0.0, 1.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.5, 0.0],
                [0.0, 1.0, 0.5],
                [0.5, 1.0, 0.0],
                [0.5, 1.0, 0.5],
                [1.0, 0.0, 0.5],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.5],
                [0.5, 1.0, 1.0],
                [1.0, 0.5, 1.0],
                [1.0, 1.0, 0.5],
            ],
            dtype=np.float64,
        ).T
    )
    elem: Type[Element] = ElementTetP2
    affine: bool = False

    @classmethod
    def init_ball(cls: Type, nrefs: int = 3) -> 'MeshTet2':
        m = MeshTet1.init_ball(nrefs=nrefs)
        M = cls.from_mesh(m)
        D = M.dofs.get_facet_dofs(M.boundary_facets()).flatten()
        doflocs = M.doflocs.copy()
        doflocs[:, D] /= np.linalg.norm(doflocs[:, D], axis=0)
        return replace(M, doflocs=doflocs)

    def _uniform(self):
        return MeshTet2.from_mesh(MeshTet1.from_mesh(self).refined())

    def _adaptive(self, marked):
        return MeshTet2.from_mesh(MeshTet1.from_mesh(self).refined(marked))
