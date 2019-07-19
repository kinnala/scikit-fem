import numpy as np

from typing import NamedTuple, Optional, Dict

from numpy import ndarray


class Dofs(NamedTuple):
    """An object containing a subset of degree-of-freedom indices."""

    nodal: Dict[str, ndarray] = {}
    facet: Dict[str, ndarray] = {}
    edge: Dict[str, ndarray] = {}
    interior: Dict[str, ndarray] = {}

    def all(self, key: Optional[str] = None):
        """Return an array consisting of all dofs.

        Parameters
        ----------
        key
            Optionally, return all dofs corresponding to a specific key.

        """

        if key is None:
            nodal = np.array([self.nodal[k] for k in self.nodal])
            facet = np.array([self.facet[k] for k in self.facet])
            edge = np.array([self.edge[k] for k in self.edge])
            interior = np.array([self.interior[k] for k in self.interior])
        else:
            if key in self.nodal:
                nodal = self.nodal[key]
            else:
                nodal = np.array([])
            if key in self.facet:
                facet = self.facet[key]
            else:
                facet = np.array([])
            if key in self.edge:
                edge = self.edge[key]
            else:
                edge = np.array([])
            if key in self.interior:
                interior = self.interior[key]
            else:
                interior = np.array([])

        return np.concatenate((
            nodal.flatten(),
            facet.flatten(),
            edge.flatten(),
            interior.flatten())).astype(np.int)
