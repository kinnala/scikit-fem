import numpy as np

from numpy import ndarray


class Topology:
    """Connectivity between abstract vertices."""

    t: ndarray = None
    _facets: ndarray = None
    _t2f: ndarray = None
    _edges: ndarray = None
    _t2e: ndarray = None

    def __init__(self, t):
        self.t = t

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
    def facets(self):
        if self._facets is None:
            self._build_facets()
        return self._facets

    @property
    def t2f(self):
        if self._t2f is None:
            self._build_facets()
        return self._t2f

    def _build_facets(self):
        self._facets, self._t2f = self._build_entities(self.facet_indices())

    @property
    def edges(self):
        if self._edges is None:
            self._build_edges()
        return self._edges

    @property
    def t2e(self):
        if self._t2e is None:
            self._build_edges()
        return self._t2e

    def _build_edges(self):
        self._edges, self._t2e = self._build_entities(self.edge_indices())

    def _build_entities(self, indices):
        """Build lower dimensional topological entities."""

        indexing = np.sort(np.hstack(
            tuple([self.t[entity] for entity in indices])
        ), axis=0)

        indexing, ixa, ixb = np.unique(indexing,
                                       axis=1,
                                       return_index=True,
                                       return_inverse=True)
        mapping = ixb.reshape((len(indices), self.nelements))

        return np.ascontiguousarray(indexing), mapping

    def edge_indices(self):
        """Define edge indexing uniquely."""

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
        raise NotImplementedError("No edge indexing for the given topology.")

    def facet_indices(self):
        """Define facet indexing uniquely."""

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
        raise NotImplementedError("No facet indexing for the given topology.")
