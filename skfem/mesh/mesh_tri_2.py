from dataclasses import dataclass, replace
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementTriP2
from .mesh_tri_1 import MeshTri1
from .mesh_2d_2 import Mesh2D2


@dataclass(repr=False)
class MeshTri2(Mesh2D2, MeshTri1):
    """A quadratic triangular mesh."""

    doflocs: ndarray = np.array([[0., 0.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 1.],
                                 [.5, 0.],
                                 [0., .5],
                                 [.5, .5],
                                 [1., .5],
                                 [.5, 1.]], dtype=np.float64).T
    elem: Type[Element] = ElementTriP2
    affine: bool = False
    sort_t: bool = False

    @classmethod
    def init_circle(cls: Type,
                    nrefs: int = 3) -> 'MeshTri2':
        m = MeshTri1.init_circle(nrefs=nrefs)
        M = cls.from_mesh(m)
        D = M.dofs.get_facet_dofs(M.boundary_facets()).flatten()
        doflocs = M.doflocs.copy()
        doflocs[:, D] /= np.linalg.norm(doflocs[:, D], axis=0)
        return replace(M, doflocs=doflocs)
