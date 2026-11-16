from dataclasses import dataclass
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

    def interior_edges(self) -> ndarray:
        """Return an array of interior edge indices."""
        return np.setdiff1d(np.arange(self.edges.shape[1], dtype=np.int32),
                            self.boundary_edges())
