import warnings

import numpy as np
from numpy import ndarray

from .geometry import Geometry


class BaseMesh:

    geom: Geometry
    boundaries = None

    def __init__(self, *args, **kwargs):
        self.geom = Geometry(*args, **kwargs)

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
        super(MeshTri1, self).__init__(2, t, ElementTriP1(), p)


class MeshQuad1(BaseMesh2D):

    def __init__(self, p=None, t=None):
        from skfem.element import ElementQuad1
        super(MeshQuad1, self).__init__(2, t, ElementQuad1(), p)


class MeshTri2(BaseMesh2D):

    def __init__(self, p=None, t=None):
        from skfem.element import ElementTriP2
        _t = t[:3]
        uniq, ix = np.unique(_t, return_inverse=True)
        rest = np.setdiff1d(np.arange(np.max(t) + 1, dtype=np.int64),
                            uniq)
        _p = p[:, uniq]
        _p = np.hstack((_p, p[:, rest]))
        __t = (np.arange(len(uniq), dtype=np.int64)[ix]
               .reshape(_t.shape))
        super(MeshTri2, self).__init__(2, __t, ElementTriP2(), _p)
        import pdb; pdb.set_trace()
        self.doflocs[:, t[3:].flatten('F')] = self.doflocs[:, self.dofs.element_dofs[3:].flatten('F')]


class MeshQuad2(BaseMesh2D):

    def __init__(self, p=None, t=None):
        from skfem.element import ElementQuad2
        super(MeshQuad2, self).__init__(2, t, ElementQuad2(), p)


class MeshTet1(BaseMesh):

    def __init__(self, p=None, t=None):
        from skfem.element import ElementTetP1
        super(MeshTet1, self).__init__(3, t, ElementTetP1(), p)


class MeshHex1(BaseMesh2D):

    def __init__(self, p=None, t=None):
        from skfem.element import ElementHex1
        super(MeshHex1, self).__init__(3, t, ElementHex1(), p)
