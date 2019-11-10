import numpy as np

from ..mesh import Mesh

from typing import Callable, Optional

from numpy import ndarray


class Mesh3D(Mesh):
    """Three dimensional meshes, common methods.

    See the following implementations:

    - :class:`~skfem.mesh.MeshTet`, tetrahedral mesh
    - :class:`~skfem.mesh.MeshHex`, hexahedral mesh

    """

    def edges_satisfying(self, test: Callable[[ndarray], bool]) -> ndarray:
        """Return edges whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            Evaluates to 1 or True for edge midpoints of the edges belonging to
            the output set.

        """
        mx = 0.5 * (self.p[0, self.edges[0, :]] + self.p[0, self.edges[1, :]])
        my = 0.5 * (self.p[1, self.edges[0, :]] + self.p[1, self.edges[1, :]])
        mz = 0.5 * (self.p[2, self.edges[0, :]] + self.p[2, self.edges[1, :]])
        return np.nonzero(test(mx, my, mz))[0]

    def boundary_edges(self) -> ndarray:
        """Return an array of boundary edge indices."""
        return np.nonzero(np.isin(self.edges,
                                  self.boundary_nodes()).all(axis=0))[0]

    def interior_edges(self) -> ndarray:
        """Return an array of interior edge indices."""
        return np.setdiff1d(np.arange(self.edges.shape[1], dtype=np.int),
                            self.boundary_edges())

    def param(self) -> float:
        """Return mesh parameter, viz the length of the longest edge."""
        lengths = np.linalg.norm(np.diff(self.p[:, self.edges], axis=1), axis=0)
        return np.max(lengths)
