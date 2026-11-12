from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy import ndarray

from .mesh import Mesh


@dataclass(repr=False)
class Mesh3D(Mesh):

    def params(self) -> ndarray:
        return np.linalg.norm(
            np.diff(self.p[:, self.edges], axis=1),
            axis=0
        )[0, self.t2e].max(axis=0)

    def param(self) -> float:
        return np.max(self.params())

    def edges_satisfying(self, test: Callable[[ndarray], bool]) -> ndarray:
        """Return edges whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            Evaluates to 1 or ``True`` for edge midpoints of the edges
            belonging to the output set.

        """
        return (np.nonzero(test(self.p[:, self.edges].mean(1)))[0]
                .astype(np.int32))

    def interior_edges(self) -> ndarray:
        """Return an array of interior edge indices."""
        return np.setdiff1d(np.arange(self.edges.shape[1], dtype=np.int32),
                            self.boundary_edges())
