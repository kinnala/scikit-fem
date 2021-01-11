from typing import Callable, Tuple

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
    facets = np.array([], dtype=np.int64)
    t2f = np.array([], dtype=np.int64)

    def _expand_facets(self, facets: ndarray) -> Tuple[ndarray, ndarray]:
        """Find vertices and edges corresponding to given facets."""
        vertices = np.unique(self.facets[:, facets].flatten())
        edge_candidates = self.t2e[:, self.f2t[0, facets]].flatten()
        # subset of edges that share all points with the given facets
        subset_ix = np.nonzero(
            np.prod(np.isin(self.edges[:, edge_candidates],
                            self.facets[:, facets].flatten()),
                    axis=0)
        )[0]
        edges = np.intersect1d(
            self.boundary_edges(),
            edge_candidates[subset_ix]
        )
        return vertices, edges

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
        ix = np.where(np.in1d(np.ravel_multi_index(A.T, dims),
                              np.ravel_multi_index(B.T, dims)))[0]
        return edge_candidates[ix]

    def interior_edges(self) -> ndarray:
        """Return an array of interior edge indices."""
        return np.setdiff1d(np.arange(self.edges.shape[1], dtype=np.int),
                            self.boundary_edges())

    def param(self) -> float:
        """Return mesh parameter, viz the length of the longest edge."""
        lengths = np.linalg.norm(
            np.diff(self.p[:, self.edges], axis=1), axis=0)
        return np.max(lengths)
