from dataclasses import dataclass, replace
from typing import Type

import numpy as np
from numpy import ndarray

from ..element import Element, ElementTriP1DG
from .mesh_tri_1 import MeshTri1


@dataclass(repr=False)
class MeshTri1DG(MeshTri1):
    """A triangular mesh with a discontinuous topology.

    The difference to :class:`~skfem.mesh.MeshTri1` is that the nodes must be
    repeated for each element.  This allows describing, e.g., periodic meshes
    while still persisting a consistent interface for visualization and
    reference mappings.

    """
    doflocs: ndarray = np.array([[0., 0.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 1.]], dtype=np.float64).T
    elem: Type[Element] = ElementTriP1DG
    affine: bool = False
    sort_t: bool = False

    def periodic(self, ix1, ix2):
        """Turn into a periodic mesh.

        Parameters
        ----------
        ix1
        ix2
        """
        assert len(ix1) == len(ix2)
        remap = np.arange(self.doflocs.shape[1], dtype=np.int64)
        remap[ix1] = ix2
        return replace(
            self,
            t=remap[self.t],
        )
