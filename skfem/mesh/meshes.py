import warnings

import numpy as np
from numpy import ndarray

from .geometry import Geometry


class BaseMesh:

    geom: Geometry

    boundaries = None

    def __init__(self, doflocs, dofs, elem, slice_dofs=None):

        warnings.warn("High-order mesh is an experimental feature and "
                      "not governed by the semantic versioning.")

        if slice_dofs is not None:
            uniq, ix = np.unique(dofs[slice_dofs], return_inverse=True)
            p = doflocs[:, uniq]
            t = (np.arange(len(uniq), dtype=np.int64)[ix]
                 .reshape(dofs[slice_dofs].shape))
            self.geom = Geometry(p, t, elem, doflocs, dofs)
        else:
            self.geom = Geometry(doflocs, dofs, elem)

    @classmethod
    def load(cls, filename):
        from skfem.io.meshio import from_file
        return from_file(filename)

    def __getattr__(self, item):
        return getattr(self.geom, item)

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p


class BaseMesh2D(BaseMesh):

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p[:, :2]


class MeshTri1(BaseMesh2D):

    def __init__(self, p=None, t=None):
        from skfem.element import ElementTriP1
        super(MeshTri1, self).__init__(p, t, ElementTriP1())


class MeshQuad1(BaseMesh2D):

    def __init__(self, p=None, t=None):
        from skfem.element import ElementQuad1
        super(MeshQuad1, self).__init__(p, t, ElementQuad1())


class MeshTri2(BaseMesh2D):

    def __init__(self, doflocs, dofs):
        from skfem.element import ElementTriP2
        super(MeshTri2, self).__init__(
            doflocs,
            dofs,
            ElementTriP2(),
            slice_dofs=slice(0, 3),
        )


class MeshQuad2(BaseMesh2D):

    def __init__(self, doflocs, dofs):
        from skfem.element import ElementQuad2
        super(MeshQuad2, self).__init__(
            doflocs,
            dofs,
            ElementQuad2(),
            slice_dofs=slice(0, 4),
        )


class MeshTet1(BaseMesh):

    def __init__(self, p=None, t=None):
        from skfem.element import ElementTetP1
        super(MeshTet1, self).__init__(p, t, ElementTetP1())


class MeshHex1(BaseMesh2D):

    def __init__(self, p=None, t=None):
        from skfem.element import ElementHex1
        super(MeshHex1, self).__init__(p, t, ElementHex1())
