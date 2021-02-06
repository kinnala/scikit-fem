from functools import lru_cache
from typing import Optional, Type

import numpy as np

from numpy import ndarray

from skfem import MappingIsoparametric, Element, ElementLineP1
from skfem.assembly import Dofs
from skfem import ElementTriP1
from skfem import *

from dataclasses import dataclass, field, replace


@dataclass
class Graph:

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
    @lru_cache(maxsize=1)
    def _facets_and_connectivity(self):
        return Graph.build_entities(self.t, self._facet_indices)

    @property
    @lru_cache(maxsize=1)
    def _edges_and_connectivity(self):
        return Graph.build_entities(self.t, self._edge_indices)

    @property
    @lru_cache(maxsize=1)
    def facets(self):
        return self._facets_and_connectivity[0]

    @property
    @lru_cache(maxsize=1)
    def t2f(self):
        return self._facets_and_connectivity[1]

    @property
    @lru_cache(maxsize=1)
    def f2t(self):
        return Graph.build_inverse(self.t, self.t2f)

    @property
    @lru_cache(maxsize=1)
    def edges(self):
        return self._edges_and_connectivity[0]

    @property
    @lru_cache(maxsize=1)
    def t2e(self):
        return self._edges_and_connectivity[1]

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

        if self.t.shape[0] == 4:
            return [
                [0, 1],
                [1, 2],
                [0, 2],
                [0, 3],
                [1, 3],
                [2, 3],
            ]
        elif self.t.shape[0] == 8:
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
        else:
            raise Exception("!")

    @property
    def _facet_indices(self):

        if self.t.shape[0] == 3:
            return [
                [0, 1],
                [1, 2],
                [0, 2],
            ]
        elif self.t.shape[0] == 4:
            return [
                [0, 1],
                [1, 2],
                [2, 3],
                [0, 3],
            ]
        elif self.t.shape[0] == 6:
            return [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ]
        elif self.t.shape[0] == 8:
            return [
                [0, 1, 4, 2],
                [0, 2, 6, 3],
                [0, 3, 5, 1],
                [2, 4, 7, 6],
                [1, 5, 7, 4],
                [3, 6, 7, 5],
            ]
        else:
            raise Exception("!")


@dataclass
class Grid(Graph):

    p: ndarray
    elem: Element

    @property
    def dofs(self):
        if not hasattr(self, '_dofs'):
            self._dofs = Dofs(self, self.elem)
        return self._dofs

    @property
    def refdom(self):  # todo
        return self.elem.mesh_type.refdom

    @property
    def brefdom(self):  # todo
        return self.elem.mesh_type.brefdom

    def _mapping(self):

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

        return MappingIsoparametric(replace(self, t=self.dofs.element_dofs),
                                    self.elem,
                                    BOUNDARY_ELEMENT_MAP[type(self.elem)]())

    def dim(self):
        return self.elem.dim
