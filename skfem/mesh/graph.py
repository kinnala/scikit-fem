from typing import Optional, Type, Tuple

import numpy as np

from numpy import ndarray

from dataclasses import dataclass, replace


@dataclass
class Graph:

    p: ndarray
    t: ndarray

    @property
    def nelements(self):
        return self.t.shape[1]

    @property
    def nvertices(self):
        return np.max(self.t) + 1

    @property
    def nfacets(self):
        return self.facets.shape[1]

    @property
    def nedges(self):
        return self.edges.shape[1]

    @property
    def nnodes(self):
        return self.t.shape[0]

    @property
    def dimension(self):
        return self.p.shape[0]

    def _init_facets(self):
        self._facets, self._t2f = Graph.build_entities(
            self.t,
            self._facet_indices
        )

    def _init_edges(self):
        self._edges, self._t2e = Graph.build_entities(
            self.t,
            self._edge_indices
        )

    @property
    def facets(self):
        if not hasattr(self, '_facets'):
            self._init_facets()
        return self._facets

    @property
    def t2f(self):
        if not hasattr(self, '_t2f'):
            self._init_facets()
        return self._t2f

    @property
    def f2t(self):
        if not hasattr(self, '_f2t'):
            self._f2t = Graph.build_inverse(self.t, self.t2f)
        return self._f2t

    @property
    def edges(self):
        if not hasattr(self, '_edges'):
            self._init_edges()
        return self._edges

    @property
    def t2e(self):
        if not hasattr(self, '_t2e'):
            self._init_edges()
        return self._t2e

    @staticmethod
    def build_entities(t, indices):

        indexing = np.sort(np.hstack(
            tuple([t[entity] for entity in indices])
        ), axis=0)

        indexing, ixa, ixb = np.unique(indexing,
                                       axis=1,
                                       return_index=True,
                                       return_inverse=True)
        mapping = ixb.reshape((len(indices), t.shape[1]))

        return np.ascontiguousarray(indexing), mapping

    @staticmethod
    def build_inverse(t, mapping):

        e = mapping.flatten(order='C')
        tix = np.tile(np.arange(t.shape[1]), (1, t.shape[0]))[0]

        e_first, ix_first = np.unique(e, return_index=True)
        e_last, ix_last = np.unique(e[::-1], return_index=True)
        ix_last = e.shape[0] - ix_last - 1

        inverse = np.zeros((2, np.max(mapping) + 1), dtype=np.int64)
        inverse[0, e_first] = tix[ix_first]
        inverse[1, e_last] = tix[ix_last]
        inverse[1, np.nonzero(inverse[0] == inverse[1])[0]] = -1

        return inverse

    @property
    def _edge_indices(self):

        if self.dimension == 3:
            if self.nnodes == 4:
                return [
                    [0, 1],
                    [1, 2],
                    [0, 2],
                    [0, 3],
                    [1, 3],
                    [2, 3],
                ]
            elif self.nnodes == 8:
                return [
                    [0, 1],
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
                    [6, 7],
                ]
        raise Exception("Cannot build edges for the given mesh.")

    @property
    def _facet_indices(self):

        if self.nnodes == 3 and self.dimension == 2:
            return [
                [0, 1],
                [1, 2],
                [0, 2],
            ]
        elif self.nnodes == 4 and self.dimension == 2:
            return [
                [0, 1],
                [1, 2],
                [2, 3],
                [0, 3],
            ]
        elif self.nnodes == 4 and self.dimension == 3:
            return [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ]
        elif self.nnodes == 8 and self.dimension == 3:
            return [
                [0, 1, 4, 2],
                [0, 2, 6, 3],
                [0, 3, 5, 1],
                [2, 4, 7, 6],
                [1, 5, 7, 4],
                [3, 6, 7, 5],
            ]
        return [[]]
        #raise Exception("Cannot build facets for the given mesh.")

    def boundary_facets(self) -> ndarray:
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1] == -1)[0]

    def boundary_edges(self) -> ndarray:
        """Return an array of boundary edge indices."""
        facets = self.boundary_facets()
        boundary_edges = np.sort(np.hstack(
            tuple([np.vstack((self.facets[itr, facets],
                              self.facets[(itr + 1) % self.facets.shape[0],
                              facets]))
                   for itr in range(self.facets.shape[0])])).T, axis=1)
        edge_candidates = np.unique(self.t2e[:, self.f2t[0, facets]])
        A = self.edges[:, edge_candidates].T
        B = boundary_edges
        dims = A.max(0) + 1
        ix = np.where(np.in1d(np.ravel_multi_index(A.T, dims),
                              np.ravel_multi_index(B.T, dims)))[0]
        return edge_candidates[ix]

    def _expand_facets(self, ix: ndarray) -> Tuple[ndarray, ndarray]:

        vertices = np.unique(self.facets[:, ix].flatten())

        if self.dimension == 3:
            edge_candidates = self.t2e[:, self.f2t[0, ix]].flatten()
            # subset of edges that share all points with the given facets
            subset = np.nonzero(
                np.prod(np.isin(self.edges[:, edge_candidates],
                                self.facets[:, ix].flatten()),
                        axis=0)
            )[0]
            edges = np.intersect1d(self.boundary_edges(),
                                   edge_candidates[subset])
        else:
            edges = np.array([], dtype=np.int64)

        return vertices, edges


@dataclass
class Grid(Graph):

    elem: 'Element'
    doflocs: Optional[ndarray] = None
    t_dofs: Optional[ndarray] = None

    @property
    def refdom(self):  # todo
        return self.elem.mesh_type.refdom

    @property
    def brefdom(self):  # todo
        return self.elem.mesh_type.brefdom

    def _mapping(self):

        from skfem.mapping import MappingIsoparametric
        from skfem.element import (ElementTriP1, ElementTriP2,
                                   ElementLineP1, ElementLineP2,
                                   ElementQuad1, ElementQuad2,
                                   ElementTetP1, ElementTetP2,
                                   ElementHex1, ElementHex2)

        BOUNDARY_ELEMENT_MAP = {
            ElementTriP1: ElementLineP1,
            ElementTriP2: ElementLineP2,
            ElementQuad1: ElementLineP1,
            ElementQuad2: ElementLineP2,
            ElementTetP1: ElementTriP1,
            ElementTetP2: ElementTriP2,
            ElementHex1: ElementQuad1,
            ElementHex2: ElementQuad2,
        }

        return MappingIsoparametric(
            replace(self,
                    t=self.t_dofs,
                    p=self.doflocs) if self.doflocs is not None else self,
            self.elem,
            BOUNDARY_ELEMENT_MAP[type(self.elem)]()
        )

    def dim(self):
        return self.elem.dim


class BaseMesh:

    grid: Grid

    def __init__(self, t, p, elem, element_dofs=None, doflocs=None):

        self.grid = Grid(
            p=p,
            t=t,
            elem=elem,
            doflocs=doflocs,
            t_dofs=element_dofs,
        )

    @classmethod
    def load(cls, filename):

        from skfem.io.meshio import from_file

        return from_file(filename)

    def __getattr__(self, item):
        return getattr(self.grid, item)


class BaseMesh2D(BaseMesh):

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p[:, :2]


class MeshTri1(BaseMesh2D):

    def __init__(self, p, t, **kwargs):

        from skfem.element import ElementTriP1

        super(MeshTri1, self).__init__(p, t, ElementTriP1())


class MeshTri2(BaseMesh2D):

    def __init__(self, doflocs, t_dofs, **kwargs):

        from skfem.element import ElementTriP2

        dofs, ix = np.unique(t_dofs[:3], return_inverse=True)
        p = doflocs[:, dofs]
        t = (np.arange(len(dofs), dtype=np.int64)[ix]
             .reshape(t_dofs[:3].shape))
        super(MeshTri2, self).__init__(
            p,
            t,
            ElementTriP2(),
            doflocs,
            t_dofs,
        )
