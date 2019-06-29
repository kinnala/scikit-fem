import numpy as np
import matplotlib.pyplot as plt

from skfem.element import ElementQuad1, ElementLineP1
from skfem.mapping import MappingIsoparametric

from .mesh2d import Mesh2D, MeshType
from .mesh_tri import MeshTri

from typing import Optional, Type, Dict

from numpy import ndarray


class MeshQuad(Mesh2D):
    """A mesh consisting of quadrilateral elements.

    The different constructors are

        - :meth:`~skfem.mesh.MeshQuad.__init__`
        - :meth:`~skfem.mesh.MeshQuad.load` (requires meshio)
        - :meth:`~skfem.mesh.MeshQuad.init_refdom`
        - :meth:`~skfem.mesh.MeshQuad.init_tensor`
    
    Attributes
    ----------
    facets
        Each column contains a pair of indices to p (2 x Nfacets).
    f2t
        Each column contains a pair of indices to t or -1 on the
        second row if the facet is on the boundary (2 x Nfacets).
    t2f
        Each column contains four indices to facets (4 x Nelements).
    
    Examples
    --------
    Initialise a tensor-product mesh.
    
    >>> from skfem.mesh import MeshQuad
    >>> import numpy as np
    >>> m = MeshQuad.init_tensor(np.linspace(0, 1, 10), np.linspace(0, 2, 5))
    >>> m.p.shape
    (2, 50)

    """

    refdom: str = "quad"
    brefdom: str = "line"
    meshio_type: str = "quad"
    name: str = "Quadrilateral"

    def __init__(self,
                 p: Optional[ndarray] = None,
                 t: Optional[ndarray] = None,
                 boundaries: Optional[Dict[str, ndarray]] = None,
                 subdomains: Optional[Dict[str, ndarray]] = None,
                 validate: Optional[bool] = True):
        """Initialise a quadrilateral mesh.

        Parameters
        ----------
        p
            The points of the mesh (2 x Nvertices).
        t
            The element connectivity (4 x Nelems), i.e. indices to p.
            These should be in counter-clockwise order.

        """
        if p is None and t is None:
            p = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]).T
            t = np.array([[0, 1, 2, 3]]).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        self.boundaries = boundaries
        self.subdomains = subdomains
        super(MeshQuad, self).__init__()
        if validate:
            self._validate()
        self._build_mappings()

    @classmethod
    def init_tensor(cls: Type[MeshType],
                    x: ndarray,
                    y: ndarray) -> MeshType:
        """Initialise a tensor product mesh.

        Parameters
        ----------
        x : numpy array (1d)
            The nodal coordinates in dimension x
        y : numpy array (1d)
            The nodal coordinates in dimension y

        """
        npx = len(x)
        npy = len(y)
        X, Y = np.meshgrid(np.sort(x), np.sort(y))   
        p = np.vstack((X.flatten('F'), Y.flatten('F')))
        ix = np.arange(npx * npy)
        nt = (npx - 1) * (npy - 1)
        t = np.zeros((4, nt))
        ix = ix.reshape(npy, npx, order='F').copy()
        t[0, :] = (ix[0:(npy-1), 0:(npx-1)].reshape(nt, 1, order='F')
                                           .copy()
                                           .flatten())
        t[1, :] = (ix[1:npy, 0:(npx-1)].reshape(nt, 1, order='F')
                                       .copy()
                                       .flatten())
        t[2, :] = (ix[1:npy, 1:npx].reshape(nt, 1, order='F')
                                   .copy()
                                   .flatten())
        t[3, :] = (ix[0:(npy-1), 1:npx].reshape(nt, 1, order='F')
                                       .copy()
                                       .flatten())
        return cls(p, t.astype(np.int64))

    @classmethod
    def init_refdom(cls: Type[MeshType]) -> MeshType:
        """Initialise a mesh that includes only the reference quad.
        
        The mesh topology is as follows::

             (-1,1) *-------------* (1,1)
                    |             |
                    |             |
                    |             |
                    |             | 
                    |             | 
                    |             |
                    |             |  
            (-1,-1) *-------------* (1,-1)

        """
        p = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]).T
        t = np.array([[0, 1, 2, 3]]).T
        return cls(p, t)

    def _build_mappings(self):
        # do not sort since order defines counterclockwise order
        # self.t=np.sort(self.t,axis=0)

        # define facets: in the order (0,1) (1,2) (2,3) (0,3)
        self.facets = np.sort(np.hstack((
            self.t[[0, 1], :],
            self.t[[1, 2], :],
            self.t[[2, 3], :],
            self.t[[0, 3], :],
        )), axis=0)

        # get unique facets and build quad-to-facet mapping: 4 (edges) x Nquads
        tmp = np.ascontiguousarray(self.facets.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)]*tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        self.facets = self.facets[:, ixa]
        self.t2f = ixb.reshape((4, self.t.shape[1]))

        # build facet-to-quadrilateral mapping: 2 (quads) x Nedges
        e_tmp = np.hstack((self.t2f[0, :],
                           self.t2f[1, :],
                           self.t2f[2, :],
                           self.t2f[3, :]))
        t_tmp = np.tile(np.arange(self.t.shape[1]), (1, 4))[0]

        e_first, ix_first = np.unique(e_tmp, return_index=True)
        # this emulates matlab unique(e_tmp,'last')
        e_last, ix_last = np.unique(e_tmp[::-1], return_index=True)
        ix_last = e_tmp.shape[0] - ix_last - 1

        self.f2t = np.zeros((2, self.facets.shape[1]), dtype=np.int64)
        self.f2t[0, e_first] = t_tmp[ix_first]
        self.f2t[1, e_last] = t_tmp[ix_last]

        # second row to -1 if repeated (i.e., on boundary)
        self.f2t[1, np.nonzero(self.f2t[0, :] == self.f2t[1, :])[0]] = -1

    def _uniform_refine(self):
        """Perform a single mesh refine that halves 'h'. Each
        quadrilateral is split into four."""
        # rename variables
        t = np.copy(self.t)
        p = np.copy(self.p)
        e = self.facets
        sz = p.shape[1]
        t2f = self.t2f + sz

        # quadrilateral middle point
        mid = range(self.t.shape[1]) + np.max(t2f) + 1
        
        # new vertices are the midpoints of edges ...
        newp1 = 0.5*np.vstack((p[0, e[0, :]] + p[0, e[1, :]],
                               p[1, e[0, :]] + p[1, e[1, :]]))

        # ... and element middle points
        newp2 = 0.25*np.vstack((p[0, t[0, :]] + p[0, t[1, :]] +
                                p[0, t[2, :]] + p[0, t[3, :]],
                                p[1, t[0, :]] + p[1, t[1, :]] +
                                p[1, t[2, :]] + p[1, t[3, :]]))
        self.p = np.hstack((p, newp1, newp2))

        # build new quadrilateral definitions
        self.t = np.hstack((
            np.vstack((t[0, :], t2f[0, :], mid, t2f[3, :])),
            np.vstack((t2f[0, :], t[1, :], t2f[1, :], mid)),
            np.vstack((mid, t2f[1, :], t[2, :], t2f[2, :])),
            np.vstack((t2f[3, :], mid, t2f[2, :], t[3, :])),
            ))

        # build mapping between old and new facets
        new_facets = np.zeros((2, e.shape[1]), dtype=np.int64)
        ix0 = np.arange(t.shape[1], dtype=np.int64)
        ix1 = ix0 + t.shape[1]
        ix2 = ix0 + 2*t.shape[1]
        ix3 = ix0 + 3*t.shape[1]

        self._build_mappings()

        new_facets[0, t2f[0, :] - sz] = self.t2f[0, ix0]
        new_facets[1, t2f[0, :] - sz] = self.t2f[0, ix1]

        new_facets[0, t2f[1, :] - sz] = self.t2f[1, ix1]
        new_facets[1, t2f[1, :] - sz] = self.t2f[1, ix2]

        new_facets[0, t2f[2, :] - sz] = self.t2f[2, ix2]
        new_facets[1, t2f[2, :] - sz] = self.t2f[2, ix3]

        new_facets[0, t2f[3, :] - sz] = self.t2f[3, ix3]
        new_facets[1, t2f[3, :] - sz] = self.t2f[3, ix0]

        self._fix_boundaries(new_facets)

    def _splitquads(self, x=None):
        """Split each quad into two triangles and return MeshTri."""
        t = self.t[[0, 1, 3], :]
        t = np.hstack((t, self.t[[1, 2, 3]]))

        if x is not None:
            if len(x) == self.t.shape[1]:
                # preserve elemental constant functions
                X = np.concatenate((x, x))
            else:
                raise Exception("The parameter x must have one " +
                                "value per element.")
            return MeshTri(self.p, t, validate=False), X
        else:
            return MeshTri(self.p, t, validate=False)

    def _splitquads_symmetric(self):
        """Split quads into four triangles."""
        t = np.vstack((self.t, np.arange(self.t.shape[1]) + self.p.shape[1]))
        newt = t[[0, 1, 4], :]
        newt = np.hstack((newt, t[[1, 2, 4], :]))
        newt = np.hstack((newt, t[[2, 3, 4], :]))
        newt = np.hstack((newt, t[[3, 0, 4], :]))
        mx = np.sum(self.p[0, self.t], axis=0) / self.t.shape[0]
        my = np.sum(self.p[1, self.t], axis=0) / self.t.shape[0]
        return MeshTri(np.hstack((self.p, np.vstack((mx, my)))),
                       newt,
                       validate=False)

    def plot(self, z, smooth=False, edgecolors=None, ax=None, zlim=None):
        """Visualise piecewise-linear or piecewise-constant function.

        The quadrilaterals are split into two triangles
        (:class:`skfem.mesh.MeshTri`) and the respective plotting function for
        the triangular mesh is used.

        """
        if len(z) == self.t.shape[-1]:
            m, z = self._splitquads(z)
        else:
            m = self._splitquads()
        return m.plot(z, smooth, ax=ax, zlim=zlim, edgecolors=edgecolors)

    def plot3(self, z, smooth=False, ax=None):
        """Visualise nodal function (3d i.e. three axes).

        The quadrilateral mesh is split into triangular mesh (MeshTri)
        and the respective plotting function for the triangular mesh is
        used.

        """
        m, z = self._splitquads(z)
        return m.plot3(z, smooth, ax=ax)

    def mapping(self):
        return MappingIsoparametric(self, ElementQuad1(), ElementLineP1())
