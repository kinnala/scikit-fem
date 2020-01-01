import numpy as np

from skfem.mesh import Mesh, MeshType

from typing import Callable, Optional

from numpy import ndarray


class Mesh2D(Mesh):
    """Two dimensional meshes, common methods.

    See the following implementations:

    - :class:`~skfem.mesh.MeshTri`, triangular mesh
    - :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh

    """

    facets: ndarray = np.array([])
    f2t: ndarray = np.array([])
    t2f: ndarray = np.array([])

    def mirror(self, a: float, b: float, c: float) -> MeshType:
        """Mirror a mesh by the line :math:`ax + by + c = 0`.  Returns a new
        :class:`~skfem.mesh.Mesh` object."""
        tmp = -2.0*(a*self.p[0, :] + b*self.p[1, :] + c) / (a**2 + b**2)
        newx = a*tmp + self.p[0, :]
        newy = b*tmp + self.p[1, :]
        newpoints = np.vstack((newx, newy))
        points = np.hstack((self.p, newpoints))
        tris = np.hstack((self.t, self.t + self.p.shape[1]))

        # remove duplicates
        tmp = np.ascontiguousarray(points.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)]*tmp.shape[1]),
                                  return_index=True,
                                  return_inverse=True)
        points = points[:, ixa]
        tris = ixb[tris]

        meshclass = type(self)

        return meshclass(points, tris)

    def param(self) -> float:
        """Return mesh parameter, viz. the length of the longest edge."""
        return np.max(np.linalg.norm(np.diff(self.p[:, self.facets], axis=1),
                                     axis=0))

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p[:, :2]
