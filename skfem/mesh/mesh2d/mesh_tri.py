import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from skfem.mapping import MappingAffine

from .mesh2d import Mesh2D, MeshType

from typing import Optional, Tuple, Type

from matplotlib.axes import Axes
from numpy import ndarray

class MeshTri(Mesh2D):
    """A mesh consisting of triangular elements.

    The different constructors are:

        - :meth:`~skfem.mesh.MeshTri.__init__`
        - :meth:`~skfem.mesh.MeshTri.load` (requires meshio)
        - :meth:`~skfem.mesh.MeshTri.init_symmetric`
        - :meth:`~skfem.mesh.MeshTri.init_sqsymmetric`
        - :meth:`~skfem.mesh.MeshTri.init_refdom`
        - :meth:`~skfem.mesh.MeshTri.init_tensor`
    
    Attributes
    ----------
    p
        An array containing the vertices of the mesh (2 x Nvertices).
    t
        An array containing the element connectivity (3 x Nelems).
    facets
        An array containing the facet vertices (2 x Nfacets).
    f2t
        An array containing the triangles next to each facet (2 x Nfacets).
        Each column contains two indices to t.  If the second row is zero then
        the facet is on the boundary.
    t2f
        An array containing the facets belonging to each triangle (3 x Nelems).
        Each column contains three indices to facets.

    Examples
    --------
    Initialise a symmetric mesh of the unit square.

    >>> m = MeshTri.init_sqsymmetric()
    >>> m.t.shape
    (3, 8)

    Facets (edges) and mappings from triangles to facets and vice versa are
    automatically constructed. In the following example we have 5 facets
    (edges).

    >>> m = MeshTri()
    >>> m.facets
    array([[0, 0, 1, 1, 2],
           [1, 2, 2, 3, 3]])
    >>> m.t2f
    array([[0, 2],
           [2, 4],
           [1, 3]])
    >>> m.f2t
    array([[ 0,  0,  1,  1,  1],
           [-1, -1,  0, -1, -1]])

    The value -1 implies that the facet (the edge) is on the boundary.

    Refine the triangular mesh of the unit square three times.

    >>> m = MeshTri()
    >>> m.refine(3)
    >>> m.p.shape
    (2, 81)

    """

    refdom = "tri"
    brefdom = "line"
    meshio_type = "triangle"

    p = np.array([])
    t = np.array([])
    facets = np.array([])
    f2t = np.array([])
    t2f = np.array([])

    def __init__(self,
                 p: Optional[ndarray] = None,
                 t: Optional[ndarray] = None,
                 validate: Optional[bool] = True,
                 sort_t: Optional[bool] = True):
        """Initialise a triangular mesh.

        If no arguments are given, initialises a mesh with the following
        topology::

            *-------------*
            |\            |
            |  \          |
            |    \        |
            |      \      |
            |        \    |
            |          \  |
            |            \|
            *-------------*

        Parameters
        ----------
        p
            An array containing the points of the mesh (2 x Nvertices).
        t
            An array containing the element connectivity (3 x Nelems), i.e.
            indices to p.
        validate
            If true, run mesh validity checks.
        sort_t
            If true, sort the element connectivity matrix before building
            mappings.

        """
        if p is None and t is None:
            p = np.array([[0., 1., 0., 1.], [0., 0., 1., 1.]], dtype=np.float_)
            t = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        if validate:
            self._validate()
        self._build_mappings(sort_t=sort_t)
        super(MeshTri, self).__init__()

    @classmethod
    def init_tensor(cls: Type[MeshType],
                    x: ndarray,
                    y: ndarray) -> MeshType:
        """Initialise a tensor product mesh.

        Parameters
        ----------
        x
            The nodal coordinates in dimension x.
        y
            The nodal coordinates in dimension y.

        """
        return MeshQuad.init_tensor(x, y)._splitquads()

    @classmethod
    def init_symmetric(cls):
        """Initialise a symmetric mesh of the unit square.
        
        The mesh topology is as follows::

            *------------*
            |\          /|
            |  \      /  |
            |    \  /    |
            |     *      |
            |    /  \    |
            |  /      \  |
            |/          \|
            *------------*

        """
        p = np.array([[0, 1, 1, 0, 0.5],
                      [0, 0, 1, 1, 0.5]], dtype=np.float_)
        t = np.array([[0, 1, 4],
                      [1, 2, 4],
                      [2, 3, 4],
                      [0, 3, 4]], dtype=np.intp).T
        return cls(p, t)

    @classmethod
    def init_sqsymmetric(cls: Type[MeshType]) -> MeshType:
        """Initialise a symmetric mesh of the unit square.
        
        The mesh topology is as follows::

            *------------*
            |\    |     /|
            |  \  |   /  |
            |    \| /    |
            |-----*------|
            |    /| \    |
            |  /  |   \  |
            |/    |     \|
            *------------*

        """
        p = np.array([[0, 0.5, 1,   0, 0.5,   1, 0, 0.5, 1],
                      [0, 0,   0, 0.5, 0.5, 0.5, 1,   1, 1]], dtype=np.float_)
        t = np.array([[0, 1, 4],
                      [1, 2, 4],
                      [2, 4, 5],
                      [0, 3, 4],
                      [3, 4, 6],
                      [4, 6, 7],
                      [4, 7, 8],
                      [4, 5, 8]], dtype=np.intp).T
        return cls(p, t)

    @classmethod
    def init_refdom(cls: Type[MeshType]) -> MeshType:
        """Initialise a mesh that includes only the reference triangle.
        
        The mesh topology is as follows::

            *
            |\           
            |  \         
            |    \       
            |      \      
            |        \    
            |          \  
            |            \ 
            *-------------*

        """
        p = np.array([[0., 1., 0.],
                      [0., 0., 1.]], dtype=np.float_)
        t = np.array([[0, 1, 2]], dtype=np.intp).T
        return cls(p, t)

    def _build_mappings(self, sort_t=True):
        # sort to preserve orientations etc.
        if sort_t:
            self.t = np.sort(self.t, axis=0)

        # define facets: in the order (0,1) (1,2) (0,2)
        self.facets = np.sort(np.vstack((self.t[0, :], self.t[1, :])), axis=0)
        self.facets = np.hstack((self.facets,
                                 np.sort(np.vstack((self.t[1, :], self.t[2, :])),
                                         axis=0)))
        self.facets = np.hstack((self.facets,
                                 np.sort(np.vstack((self.t[0, :], self.t[2, :])),
                                         axis=0)))

        # get unique facets and build triangle-to-facet
        # mapping: 3 (edges) x Ntris
        tmp = np.ascontiguousarray(self.facets.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        self.facets = self.facets[:, ixa]
        self.t2f = ixb.reshape((3, self.t.shape[1]))

        # build facet-to-triangle mapping: 2 (triangles) x Nedges
        e_tmp = np.hstack((self.t2f[0, :], self.t2f[1, :], self.t2f[2, :]))
        t_tmp = np.tile(np.arange(self.t.shape[1]), (1, 3))[0]

        e_first, ix_first = np.unique(e_tmp, return_index=True)
        # this emulates matlab unique(e_tmp,'last')
        e_last, ix_last = np.unique(e_tmp[::-1], return_index=True)
        ix_last = e_tmp.shape[0] - ix_last - 1

        self.f2t = np.zeros((2, self.facets.shape[1]), dtype=np.int64)
        self.f2t[0, e_first] = t_tmp[ix_first]
        self.f2t[1, e_last] = t_tmp[ix_last]

        # second row to zero if repeated (i.e., on boundary)
        self.f2t[1, np.nonzero(self.f2t[0, :] == self.f2t[1, :])[0]] = -1

    def plot(self,
             z: ndarray,
             smooth: Optional[bool] = False,
             ax: Optional[Axes] = None,
             zlim: Optional[Tuple[float, float]] = None,
             edgecolors: Optional[str] = None,
             aspect: float = 1.) -> Axes:
        """Visualise piecewise-linear or piecewise-constant function, 2D plot.
        
        Parameters
        ----------
        z
            An array of nodal values (Nvertices) or elemental values (Nelems).
        smooth
            If true, use gouraud shading.
        ax
            Plot onto the given preinitialised Matplotlib axes.
        zlim
            Use the given minimum and maximum values for coloring.
        edgecolors
            A string describing the edge coloring, e.g. 'k' for black.
        float
            The ratio of vertical to horizontal length-scales; ignored if ax
            specified.

        Returns
        -------
        Axes
            The Matplotlib axes onto which the mesh was plotted.

        Examples
        --------
        Mesh the unit square :math:`(0,1)^2` and visualise the function
        :math:`f(x)=x^2`.

        >>> from skfem.mesh import MeshTri
        >>> m = MeshTri()
        >>> m.refine(3)
        >>> ax = m.plot(m.p[0, :]**2, smooth=True)
        >>> m.show()
            
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect(aspect)
        if edgecolors is None:
            edgecolors = 'k'
        if zlim == None:
            if smooth:
                ax.tripcolor(self.p[0, :], self.p[1, :], self.t.T, z,
                              shading='gouraud', edgecolors=edgecolors)
            else:
                ax.tripcolor(self.p[0, :], self.p[1, :], self.t.T, z, edgecolors=edgecolors)
        else:
            if smooth:
                ax.tripcolor(self.p[0, :], self.p[1, :], self.t.T, z,
                              shading='gouraud', vmin=zlim[0], vmax=zlim[1], edgecolors=edgecolors)
            else:
                ax.tripcolor(self.p[0, :], self.p[1, :], self.t.T, z,
                              vmin=zlim[0], vmax=zlim[1], edgecolors=edgecolors)
        return ax

    def plot3(self,
              z: ndarray,
              ax: Optional[Axes] = None) -> Axes:
        """Visualise piecewise-linear or piecewise-constant function, 3D plot.
        
        Parameters
        ----------
        z
            An array of nodal values (Nvertices), elemental values (Nelems)
            or three elemental values (3 x Nelems, piecewise linear DG).
        ax
            Plot onto the given preinitialised Matplotlib axes.

        Returns
        -------
        Axes
            The Matplotlib axes onto which the mesh was plotted.
        
        Examples
        --------
        Mesh the unit square :math:`(0,1)^2` and visualise the function
        :math:`f(x)=x^2`.

        >>> from skfem.mesh import MeshTri
        >>> m = MeshTri()
        >>> m.refine(3)
        >>> ax = m.plot3(m.p[1, :]**2)
        >>> m.show()
        
        """
        from mpl_toolkits.mplot3d import Axes3D
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        if len(z) == self.p.shape[1]:
            # use matplotlib
            ax.plot_trisurf(self.p[0, :], self.p[1, :], z,
                            triangles=self.t.T,
                            cmap=plt.cm.viridis)
        elif len(z) == self.t.shape[1]:
            # one value per element (piecewise const)
            nt = self.t.shape[1]
            newt = np.arange(3*nt, dtype=np.int64).reshape((nt, 3))
            newpx = self.p[0, self.t].flatten(order='F')
            newpy = self.p[1, self.t].flatten(order='F')
            newz = np.vstack((z, z, z)).flatten(order='F')
            ax.plot_trisurf(newpx, newpy, newz,
                            triangles=newt.T,
                            cmap=plt.cm.viridis)
        elif len(z) == 3*self.t.shape[1]:
            # three values per element (piecewise linear)
            nt = self.t.shape[1]
            newt = np.arange(3*nt, dtype=np.int64).reshape((nt, 3))
            newpx = self.p[0, self.t].flatten(order='F')
            newpy = self.p[1, self.t].flatten(order='F')
            ax.plot_trisurf(newpx, newpy, z,
                            triangles=newt.T,
                            cmap=plt.cm.viridis)
        else:
            raise NotImplementedError("MeshTri.plot3: not implemented for "
                                      "the given shape of input vector!")
        return ax

    def _uniform_refine(self):
        """Perform a single mesh refine."""
        # rename variables
        t = self.t
        p = self.p
        e = self.facets
        sz = p.shape[1]
        t2f = self.t2f + sz
        # new vertices are the midpoints of edges
        newp = 0.5*np.vstack((p[0, e[0, :]] + p[0, e[1, :]],
                              p[1, e[0, :]] + p[1, e[1, :]]))
        newp = np.hstack((p, newp))
        # build new triangle definitions
        newt = np.vstack((t[0, :], t2f[0, :], t2f[2, :]))
        newt = np.hstack((newt, np.vstack((t[1, :], t2f[0, :], t2f[1, :]))))
        newt = np.hstack((newt, np.vstack((t[2, :], t2f[2, :], t2f[1, :]))))
        newt = np.hstack((newt, np.vstack((t2f[0, :], t2f[1, :], t2f[2, :]))))
        # update fields
        self.p = newp
        self.t = newt

        self._build_mappings()

        # prolongation matrix
        nsz = newp.shape[1]
        return coo_matrix(
            (np.hstack((np.ones(sz), .5 * np.ones(2 * (nsz - sz)))),
             (np.hstack((np.arange(sz), np.arange(nsz - sz) + sz, np.arange(nsz - sz) + sz)),
              np.hstack((np.arange(sz), e[0, :], e[1, :])))),
            shape=(nsz, sz)).tocsr()

    def mapping(self):
        return MappingAffine(self)

    def element_finder(self):
        from matplotlib.tri import Triangulation

        return Triangulation(self.p[0, :],
                             self.p[1, :],
                             self.t.T).get_trifinder()
