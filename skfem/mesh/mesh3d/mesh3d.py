from typing import Callable

import numpy as np
from numpy import ndarray

from ..mesh import Mesh


class Mesh3D(Mesh):
    """Three dimensional meshes, common methods.

    See the following implementations:

    - :class:`~skfem.mesh.MeshTet`, tetrahedral mesh
    - :class:`~skfem.mesh.MeshHex`, hexahedral mesh

    """
    p = np.zeros((3, 0), dtype=np.float64)
    f2t = np.zeros((2, 0), dtype=np.int64)

    def edges_satisfying(self, test: Callable[[ndarray], bool]) -> ndarray:
        """Return edges whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            Evaluates to 1 or True for edge midpoints of the edges belonging to
            the output set.

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
        return np.nonzero((self.edges.T[:, None] == boundary_edges)
                          .all(-1).any(-1))[0]

    def interior_edges(self) -> ndarray:
        """Return an array of interior edge indices."""
        return np.setdiff1d(np.arange(self.edges.shape[1], dtype=np.int),
                            self.boundary_edges())

    def param(self) -> float:
        """Return mesh parameter, viz the length of the longest edge."""
        lengths = np.linalg.norm(
            np.diff(self.p[:, self.edges], axis=1), axis=0)
        return np.max(lengths)
