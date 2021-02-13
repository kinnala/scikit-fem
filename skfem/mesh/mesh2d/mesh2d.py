import warnings

import numpy as np
from numpy import ndarray

from skfem.mesh import Mesh


class Mesh2D(Mesh):
    """Two dimensional meshes, common methods.

    See the following implementations:

    - :class:`~skfem.mesh.MeshTri`, triangular mesh
    - :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh

    """
    p = np.zeros((2, 0), dtype=np.float64)
    facets = np.zeros((2, 0), dtype=np.int64)
    f2t = np.zeros((2, 0), dtype=np.int64)
    t2f = np.array([], dtype=np.int64)

    def __init__(self, *args, **kwargs):
        super(Mesh2D, self).__init__()

    def mirror(self, a: float, b: float, c: float) -> Mesh:
        """Mirror a mesh by the line :math:`ax + by + c = 0`.

        Deprecated in favour of :meth:`~skfem.mesh.Mesh.mirrored`.

        """
        warnings.warn("This method is deprecated in favour of mirrored",
                      DeprecationWarning)
        tmp = -2. * (a * self.p[0] + b * self.p[1] + c) / (a ** 2 + b ** 2)
        newx = a * tmp + self.p[0]
        newy = b * tmp + self.p[1]
        newpoints = np.vstack((newx, newy))
        points = np.hstack((self.p, newpoints))
        tris = np.hstack((self.t, self.t + self.p.shape[1]))

        # remove duplicates
        tmp = np.ascontiguousarray(points.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
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

    def _repr_svg_(self) -> Optional[str]:
        from skfem.visuals.svg import draw
        if self.t.shape[1] > 5000:
            return
        return draw(self)

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p[:, :2]
