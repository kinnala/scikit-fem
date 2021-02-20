import warnings
from typing import Type

import numpy as np
from numpy import ndarray

from .geometry import Geometry
from ..element import (Element, ElementTriP1, ElementQuad1, ElementTriP2,
                       ElementQuad2, ElementTetP1, ElementHex1)


class BaseMesh:

    geom: Geometry
    boundaries = None
    _elem: Type

    def __init__(self, p=None, t=None):

        if t.shape[0] > self._elem.refdom.nnodes:
            m = self._elem.refdom.nnodes
            _t = t[:m]
            uniq, ix = np.unique(_t, return_inverse=True)
            rest = np.setdiff1d(np.arange(np.max(t) + 1, dtype=np.int64),
                                uniq)
            _p = p[:, uniq]
            p = np.hstack((_p, p[:, rest]))
            __t = (np.arange(len(uniq), dtype=np.int64)[ix]
                   .reshape(_t.shape))
        else:
            __t = t
        self.geom = Geometry(
            self._elem.refdom.dim(),
            __t,
            self._elem(),
            p,
        )
        if t.shape[0] > self._elem.refdom.nnodes:
            self.doflocs[:, t[m:].flatten('F')] =\
                self.doflocs[:, self.dofs.element_dofs[m:].flatten('F')]

    @classmethod
    def load(cls, filename):
        from skfem.io.meshio import from_file
        return from_file(filename)

    @classmethod
    def init_refdom(cls):
        return cls(*cls._elem.refdom.init_refdom())

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

    _elem = ElementTriP1


class MeshQuad1(BaseMesh2D):

    _elem = ElementQuad1


class MeshTri2(BaseMesh2D):

    _elem = ElementTriP2

    @classmethod
    def init_refdom(cls):
        return MeshTri1(*MeshTri1._elem.refdom.init_refdom())


class MeshQuad2(BaseMesh2D):

    _elem = ElementQuad2


class MeshTet1(BaseMesh):

    _elem = ElementTetP1


class MeshHex1(BaseMesh2D):

    _elem = ElementHex1
