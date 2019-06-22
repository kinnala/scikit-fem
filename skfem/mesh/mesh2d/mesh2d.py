import numpy as np
import matplotlib.pyplot as plt

from skfem.mesh import Mesh, MeshType

from typing import Callable, Optional

from numpy import ndarray
from matplotlib.axes import Axes


class Mesh2D(Mesh):
    """Two dimensional meshes, common methods.

    See the following implementations:

    - :class:`~skfem.mesh.MeshTri`, triangular mesh
    - :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh

    """

    facets: ndarray = np.array([])
    f2t: ndarray = np.array([])
    t2f: ndarray = np.array([])

    def draw(self,
             ax: Optional[Axes] = None,
             node_numbering: Optional[bool] = False,
             facet_numbering: Optional[bool] = False,
             element_numbering: Optional[bool] = False,
             aspect: float = 1.) -> Axes:
        """Visualise a mesh by drawing the edges.

        Parameters
        ----------
        ax
            A preinitialised Matplotlib axes for plotting.
        node_numbering
            If true, draw node numbering.
        facet_numbering
            If true, draw facet numbering.
        element_numbering
            If true, draw element numbering.
        aspect
            Ratio of vertical to horizontal length-scales; ignored if ax
            specified

        Returns
        -------
        Axes
            The Matplotlib axes onto which the mesh was plotted.

        """
        if ax is None:
            # create new figure
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect(aspect)
            ax.set_axis_off()
        # visualize the mesh faster plotting is achieved through
        # None insertion trick.
        xs = []
        ys = []
        for s, t, u, v in zip(self.p[0, self.facets[0, :]],
                              self.p[1, self.facets[0, :]],
                              self.p[0, self.facets[1, :]],
                              self.p[1, self.facets[1, :]]):
            xs.append(s)
            xs.append(u)
            xs.append(None)
            ys.append(t)
            ys.append(v)
            ys.append(None)
        ax.plot(xs, ys, 'k', linewidth='0.5')

        if node_numbering:
            for itr in range(self.p.shape[1]):
                ax.text(self.p[0, itr], self.p[1, itr], str(itr))

        if facet_numbering:
            mx = .5*(self.p[0, self.facets[0, :]] +
                     self.p[0, self.facets[1, :]])
            my = .5*(self.p[1, self.facets[0, :]] +
                     self.p[1, self.facets[1, :]])
            for itr in range(self.facets.shape[1]):
                ax.text(mx[itr], my[itr], str(itr))

        if element_numbering:
            mx = np.sum(self.p[0, self.t], axis=0) / self.t.shape[0]
            my = np.sum(self.p[1, self.t], axis=0) / self.t.shape[0]
            for itr in range(self.t.shape[1]):
                ax.text(mx[itr], my[itr], str(itr))

        return ax

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
