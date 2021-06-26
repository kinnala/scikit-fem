from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy import ndarray

from .mesh import Mesh


@dataclass(repr=False)
class Mesh3D(Mesh):

    def param(self) -> float:
        return np.max(
            np.linalg.norm(np.diff(self.p[:, self.edges], axis=1), axis=0)
        )

    def edges_satisfying(self, test: Callable[[ndarray], bool]) -> ndarray:
        """Return edges whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            Evaluates to 1 or ``True`` for edge midpoints of the edges
            belonging to the output set.

        """
        return np.nonzero(test(self.p[:, self.edges].mean(1)))[0]

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
        ix = np.where(np.in1d(
            np.ravel_multi_index(A.T, dims),  # type: ignore
            np.ravel_multi_index(B.T, dims),  # type: ignore
        ))[0]
        return edge_candidates[ix]

    def interior_edges(self) -> ndarray:
        """Return an array of interior edge indices."""
        return np.setdiff1d(np.arange(self.edges.shape[1], dtype=np.int64),
                            self.boundary_edges())
