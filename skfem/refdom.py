from typing import Type, Optional, List

import numpy as np

from numpy import ndarray


class Refdom:
    """A finite element reference domain."""

    p: ndarray
    t: ndarray
    facets: Optional[List[List[int]]] = None
    edges: Optional[List[List[int]]] = None
    brefdom: Optional[Type] = None
    nnodes: int = 0
    nfacets: int = 0
    nedges: int = 0
    name: str = "Unknown"

    @classmethod
    def init_refdom(cls):
        return cls.p, cls.t

    @classmethod
    def dim(cls):
        return cls.p.shape[0]

    @classmethod
    def on_facet(cls, i, X):
        raise NotImplementedError


class RefPoint(Refdom):

    p = np.array([[0.]], dtype=np.float64)
    t = np.array([[0]], dtype=np.int64)
    name = "Zero-dimensional"


class RefLine(Refdom):

    p = np.array([[0., 1.]], dtype=np.float64)
    t = np.array([[0], [1]], dtype=np.int64)
    normals = np.array([[-1.],
                        [1.]])
    facets = [[0],
              [1]]
    brefdom = RefPoint
    nnodes = 2
    nfacets = 2
    name = "One-dimensional"


class RefTri(Refdom):

    p = np.array([[0., 1., 0.],
                  [0., 0., 1.]], dtype=np.float64)
    t = np.array([[0], [1], [2]], dtype=np.int64)
    normals = np.array([[0., -1.],
                        [1., 1.],
                        [-1., 0.]])
    facets = [[0, 1],
              [1, 2],
              [0, 2]]
    brefdom = RefLine
    nnodes = 3
    nfacets = 3
    name = "Triangular"

    @classmethod
    def on_facet(cls, i, X):
        if i == 0:
            return ((X[0] > 0)
                    * (X[0] < 1)
                    * (X[1] < 1e-4)
                    * (X[1] > -1e-4))
        elif i == 1:
            return ((X[0] > 0)
                    * (X[0] < 1)
                    * (X[0] + X[1] - 1 < 1e-4)
                    * (X[0] + X[1] - 1 > -1e-4))
        elif i == 2:
            return ((X[1] > 0)
                    * (X[1] < 1)
                    * (X[0] < 1e-4)
                    * (X[0] > -1e-4))
        raise ValueError


class RefTet(Refdom):

    p = np.array([[0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float64)
    t = np.array([[0], [1], [2], [3]], dtype=np.int64)
    facets = [[0, 1, 2],
              [0, 1, 3],
              [0, 2, 3],
              [1, 2, 3]]
    edges = [[0, 1],
             [1, 2],
             [0, 2],
             [0, 3],
             [1, 3],
             [2, 3]]
    brefdom = RefTri
    nnodes = 4
    nfacets = 4
    nedges = 6
    name = "Tetrahedral"


class RefQuad(Refdom):

    p = np.array([[0., 1., 1., 0.],
                  [0., 0., 1., 1.]], dtype=np.float64)
    t = np.array([[0], [1], [2], [3]], dtype=np.int64)
    normals = np.array([[0., -1.],
                        [1., 0.],
                        [0., 1.],
                        [-1., 0.]])
    facets = [[0, 1],
              [1, 2],
              [2, 3],
              [0, 3]]
    brefdom = RefLine
    nnodes = 4
    nfacets = 4
    name = "Quadrilateral"


class RefHex(Refdom):

    p = np.array([[1., 1., 1.],
                  [1., 1., 0.],
                  [1., 0., 1.],
                  [0., 1., 1.],
                  [1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.],
                  [0., 0., 0.]], dtype=np.float64).T
    t = np.array([[0], [1], [2], [3], [4], [5], [6], [7]], dtype=np.int64)
    normals = np.array([[1., 0., 0.],
                        [0., 0., 1.],
                        [0., 1., 0.],
                        [0., -1., 0.],
                        [0., 0., -1.],
                        [-1., 0., 0.]])
    facets = [[0, 1, 4, 2],
              [0, 2, 6, 3],
              [0, 3, 5, 1],
              [2, 4, 7, 6],
              [1, 5, 7, 4],
              [3, 6, 7, 5]]
    edges = [[0, 1],
             [0, 2],
             [0, 3],
             [1, 4],
             [1, 5],
             [2, 4],
             [2, 6],
             [3, 5],
             [3, 6],
             [4, 7],
             [5, 7],
             [6, 7]]
    brefdom = RefQuad
    nnodes = 8
    nfacets = 6
    nedges = 12
    name = "Hexahedral"


class RefWedge(Refdom):

    p = np.array([[0., 0., 0.],
                  [1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.],
                  [1., 0., 1.],
                  [0., 1., 1.]], dtype=np.float64).T
    t = np.array([[0], [1], [2], [3], [4], [5]], dtype=np.int64)
    normals = np.array([[0., -1., 0.],
                        [1., 1., 0.],
                        [-1., 0., 0.],
                        [0., 0., -1.],
                        [0., 0., 1.]])
    facets = [[0, 1, 4, 3],
              [1, 2, 5, 4],
              [0, 2, 5, 3],
              [0, 1, 2, 0],  # last index repeated
              [3, 4, 5, 3]]  # last index repeated
    edges = [[0, 1],
             [1, 2],
             [0, 2],
             [3, 4],
             [4, 5],
             [3, 5],
             [0, 3],
             [1, 4],
             [2, 5]]
    brefdom = None
    nnodes = 6
    nfacets = 5
    nedges = 9
    name = "Wedge"
