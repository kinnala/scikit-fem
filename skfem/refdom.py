from typing import Optional

import numpy as np

from numpy import ndarray


class Refdom:

    p: ndarray
    t: ndarray
    nnodes: int = 0
    nfacets: int = 0
    nedges: int = 0

    @classmethod
    def init_refdom(cls):
        return cls.p, cls.t

    @classmethod
    def dim(cls):
        return cls.p.shape[0]


class RefPoint(Refdom):

    p = np.array([[0.]], dtype=np.float_)
    t = np.array([[0]], dtype=np.int64)
    brefdom = None


class RefLine(Refdom):

    p = np.array([[0., 1.]], dtype=np.float_)
    t = np.array([[0], [1]], dtype=np.int64)
    brefdom = RefPoint
    nnodes = 2


class RefTri(Refdom):

    p = np.array([[0., 1., 0.],
                  [0., 0., 1.]], dtype=np.float_)
    t = np.array([[0], [1], [2]], dtype=np.int64)
    brefdom = RefLine
    nnodes = 3
    nfacets = 3


class RefTet(Refdom):

    p = np.array([[0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float_)
    t = np.array([[0], [1], [2], [3]], dtype=np.int64)
    brefdom = RefTri
    nnodes = 4
    nfacets = 4
    nedges = 6


class RefQuad(Refdom):

    p = np.array([[0., 1., 1., 0.],
                  [0., 0., 1., 1.]], dtype=np.float_)
    t = np.array([[0], [1], [2], [3]], dtype=np.int64)
    brefdom = RefLine
    nnodes = 4
    nfacets = 4


class RefHex(Refdom):

    p = np.array([[0., 0., 0., 1., 0., 1., 1., 1.],
                  [0., 0., 1., 0., 1., 0., 1., 1.],
                  [0., 1., 0., 0., 1., 1., 0., 1.]], dtype=np.float_)
    t = np.array([[0], [1], [2], [3], [4], [5], [6], [7]], dtype=np.int64)
    brefdom = RefQuad
    nnodes = 8
    nfacets = 6
    nedges = 12
