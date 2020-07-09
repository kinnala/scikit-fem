import warnings
from typing import NamedTuple, Dict, Union, List

import numpy as np
from numpy import ndarray


class Dofs(NamedTuple):
    """An object containing a subset of degree-of-freedom indices."""

    nodal: Dict[str, ndarray] = {}
    facet: Dict[str, ndarray] = {}
    edge: Dict[str, ndarray] = {}
    interior: Dict[str, ndarray] = {}

    def all(self, key: Union[List[str], str] = None):
        """Return an array consisting of all degrees-of-freedom.

        Parameters
        ----------
        key
            Optionally, return all degrees-of-freedom corresponding to a
            specific 'key'.

        Returns
        -------
        ndarray
            A list of degree-of-freedom indices.

        """
        if isinstance(key, list):
            return np.concatenate(tuple([self.all(k) for k in key]))

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

        output = np.concatenate((
            nodal.flatten(),
            facet.flatten(),
            edge.flatten(),
            interior.flatten()
        )).astype(np.int)

        if len(output) == 0:
            warnings.warn("Given DOF name not found in Basis. "
                          "Returning an empty set of DOF's.")

        return output

    @staticmethod
    def _merge_dicts(d1, d2):
        keys = set(d1).union(d2)
        no = np.array([])
        return dict((k, np.union1d(d1.get(k, no), d2.get(k, no)))
                    for k in keys)

    def __or__(self, other: Dofs):
        """For merging two sets of DOF's."""
        return Dofs(
            nodal=self._merge_dicts(self.nodal, other.nodal),
            facet=self._merge_dicts(self.facet, other.facet),
            edge=self._merge_dicts(self.edge, other.edge),
            interior=self._merge_dicts(self.interior, other.interior),
        )
