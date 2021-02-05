from typing import Optional

import numpy as np

from numpy import ndarray

from skfem import MappingIsoparametric, Element, ElementLineP1
from skfem.assembly import Dofs
from skfem import ElementTriP1

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
    dofs: Dofs = field(init=False)

    def __post_init__(self):
        self.dofs = Dofs(self, self.elem)

    @property
    def refdom(self):
        return "tri"

    @property
    def brefdom(self):
        return "line"

    def _mapping(self):
        return MappingIsoparametric(replace(self, t=self.dofs.element_dofs),
                                    self.elem)

    def dim(self):
        return self.elem.dim
