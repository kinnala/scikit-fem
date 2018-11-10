import numpy as np
import matplotlib.pyplot as plt

from skfem.mesh import Mesh, MeshType

from typing import Callable, Dict, Optional, Union

from numpy import ndarray
from matplotlib.axes import Axes

class Mesh2D(Mesh):
    """Two dimensional meshes, common methods.
    
    See the following implementations:

    - :class:`~skfem.mesh.MeshTri`, triangular mesh
    - :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh

    """

    def nodes_satisfying(self, test: Callable[[float, float], bool]) -> ndarray:
        """Return nodes that satisfy some condition.

        Parameters
        ----------
        test
            A function which returns True for the set of nodes that are to be
            included in the return set.

        """
        return np.nonzero(test(self.p[0, :], self.p[1, :]))[0]

    def facets_satisfying(self, test: Callable[[float, float], bool]) -> ndarray:
        """Return facets whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            A function which returns True for the facet midpoints that are to
            be included in the return set.

        """
        mx = 0.5*(self.p[0, self.facets[0, :]] + self.p[0, self.facets[1, :]])
        my = 0.5*(self.p[1, self.facets[0, :]] + self.p[1, self.facets[1, :]])
        return np.nonzero(test(mx, my))[0]

    def elements_satisfying(self, test: Callable[[float, float], bool]) -> ndarray:
        """Return elements whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            A function which returns True for the element midpoints that are to
            be included in the return set.

        """
        mx = np.sum(self.p[0, self.t], axis=0)/self.t.shape[0]
        my = np.sum(self.p[1, self.t], axis=0)/self.t.shape[0]
        return np.nonzero(test(mx, my))[0]    

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
            mx = .5*(self.p[0, self.facets[0, :]] + self.p[0, self.facets[1, :]])
            my = .5*(self.p[1, self.facets[0, :]] + self.p[1, self.facets[1, :]])
            for itr in range(self.facets.shape[1]):
                ax.text(mx[itr], my[itr], str(itr))

        if element_numbering:
            mx = np.sum(self.p[0, self.t], axis=0)/self.t.shape[0]
            my = np.sum(self.p[1, self.t], axis=0)/self.t.shape[0]
            for itr in range(self.t.shape[1]):
                ax.text(mx[itr], my[itr], str(itr))

        return ax

    def mirror(self, a: float, b: float, c: float) -> MeshType:
        """Mirror a mesh by the line :math:`ax + by + c = 0`.  Returns a new
        :class:`~skfem.mesh.Mesh` object."""
        tmp = -2.0*(a*self.p[0, :] + b*self.p[1, :] + c)/(a**2 + b**2)
        newx = a*tmp + self.p[0, :]
        newy = b*tmp + self.p[1, :]
        newpoints = np.vstack((newx, newy))
        points = np.hstack((self.p, newpoints))
        tris = np.hstack((self.t, self.t + self.p.shape[1]))

        # remove duplicates
        tmp = np.ascontiguousarray(points.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)]*tmp.shape[1]), return_index=True, return_inverse=True)
        points = points[:, ixa]
        tris = ixb[tris]

        meshclass = type(self)

        return meshclass(points, tris)

    def save(self,
            filename: str,
            pointData: Optional[Union[ndarray, Dict[str, ndarray]]] = None,
            cellData: Optional[Union[ndarray, Dict[str, ndarray]]] = None) -> None:
        """Export the mesh and fields using meshio. (2D version.)

        Parameters
        ----------
        filename
            The filename for vtk-file.
        pointData
            Data related to the vertices of the mesh. Numpy array for one
            output or dict for multiple.
        cellData
            Data related to the elements of the mesh. Numpy array for one
            output or dict for multiple

        """
        import meshio

        if pointData is not None:
            if type(pointData) != dict:
                pointData = {'0':pointData}

        if cellData is not None:
            if type(cellData) != dict:
                cellData = {'0':cellData}

        cells = { self.meshio_type : self.t.T }
        mesh = meshio.Mesh(self.p.T, cells, pointData, cellData)
        meshio.write(filename, mesh)

    def param(self) -> float:
        """Return mesh parameter."""
        return np.max(np.sqrt(np.sum((self.p[:, self.facets[0, :]] -
                                      self.p[:, self.facets[1, :]])**2, axis=0)))
    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p[:, :2]
