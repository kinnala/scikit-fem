import numpy as np

from typing import NamedTuple, Optional, Dict

from numpy import ndarray


class Dofs(NamedTuple):
    """An object containing a subset of degree-of-freedom indices."""
    nodal: Dict[str, ndarray] = {}
    facet: Dict[str, ndarray] = {}
    edge: Dict[str, ndarray] = {}
    interior: Dict[str, ndarray] = {}

    def all(self):
        """Return an array consisting of all dofs."""
        nodal = np.array([self.nodal[key] for key in self.nodal]).flatten()
        facet = np.array([self.facet[key] for key in self.facet]).flatten()
        edge = np.array([self.edge[key] for key in self.edge]).flatten()
        interior = np.array([self.interior[key] for key in self.interior]).flatten()
        return np.concatenate((nodal, facet, edge, interior)).astype(np.int)
