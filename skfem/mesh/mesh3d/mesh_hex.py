from typing import Dict, Optional, Type, Union

import numpy as np
from numpy import ndarray

from skfem.element import ElementHex1, ElementQuad1
from skfem.mapping import MappingIsoparametric

from ..mesh import MeshType
from .mesh3d import Mesh3D


class MeshHex(Mesh3D):
    """A mesh consisting of hexahedral elements.

    Attributes
    ----------
    p : numpy array of size 3 x Nvertices
        The vertices of the mesh. Each column corresponds to a point.
    t : numpy array of size 8 x Nelements
        The element connectivity. Each column corresponds to a element
        and contains eight column indices to MeshHex.p.
    facets : numpy array of size 4 x Nfacets
        Each column contains four column indices to MeshHex.p.
    f2t : numpy array of size 2 x Nfacets
        Each column contains a pair of column indices to MeshHex.t
        or -1 on the second row if the corresponding
        facet is located on the boundary.
    t2f : numpy array of size 6 x Nelements
        Each column contains four indices to MeshHex.facets.
    edges : numpy array of size 2 x Nedges
        Each column corresponds to an edge and contains two indices to
        MeshHex.p.
    t2e : numpy array of size 12 x Nelements
        Each column contains twelve column indices of MeshHex.edges.

    """

    refdom = "hex"
    brefdom = "quad"
    meshio_type = "hexahedron"

    def __init__(self, p=None, t=None, validate=True):
        """Initialise a hexahedral mesh."""
        if p is None and t is None:
            p = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.],
                          [0., 1., 1.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]]).T
            t = np.array([[0, 1, 2, 3, 4, 5, 6, 7]]).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        #
        # TODO fix orientation if p and t is provided. refer to
        # the default mesh for correct orientation
        #
        #   2---6
        #  /   /|
        # 4---7 3
        # |   |/
        # 1---5
        #
        # The hidden node is 0.
        #
        self.p = p
        self.t = t
        if validate:
            self._validate()
        self._build_mappings()
        super(MeshHex, self).__init__()

    @classmethod
    def init_tensor(cls: Type[MeshType],
                    x: ndarray,
                    y: ndarray,
                    z: ndarray) -> MeshType:
        """Initialise a tensor product mesh.

        Parameters
        ----------
        x : numpy array (1d)
            The nodal coordinates in dimension x
        y : numpy array (1d)
            The nodal coordinates in dimension y
        z : numpy array (1d)
            The nodal coordinates in dimension z

        """
        npx = len(x)
        npy = len(y)
        npz = len(z)
        X, Y, Z = np.meshgrid(np.sort(x), np.sort(y), np.sort(z))   
        p = np.vstack((X.flatten('F'), Y.flatten('F'), Z.flatten('F')))
        ix = np.arange(npx*npy*npz)
        ne = (npx-1)*(npy-1)*(npz-1)
        t = np.zeros((8, ne))
        ix = ix.reshape(npy, npx, npz, order='F').copy()
        t[0, :] = ix[0:(npy-1), 0:(npx-1), 0:(npz-1)].reshape(ne, 1, order='F').copy().flatten()
        t[1, :] = ix[1:npy, 0:(npx-1), 0:(npz-1)].reshape(ne, 1, order='F').copy().flatten()
        t[2, :] = ix[0:(npy-1), 1:npx, 0:(npz-1)].reshape(ne, 1, order='F').copy().flatten()
        t[3, :] = ix[0:(npy-1), 0:(npx-1), 1:npz].reshape(ne, 1, order='F').copy().flatten()
        t[4, :] = ix[1:npy, 1:npx, 0:(npz-1)].reshape(ne, 1, order='F').copy().flatten()
        t[5, :] = ix[1:npy, 0:(npx-1), 1:npz].reshape(ne, 1, order='F').copy().flatten()
        t[6, :] = ix[0:(npy-1), 1:npx, 1:npz].reshape(ne, 1, order='F').copy().flatten()
        t[7, :] = ix[1:npy, 1:npx, 1:npz].reshape(ne, 1, order='F').copy().flatten()
        return cls(p, t.astype(np.int64))

    def _build_mappings(self):
        """Build element-to-facet, element-to-edges, etc. mappings."""
        self.edges = np.sort(np.vstack((self.t[0, :], self.t[1, :])), axis=0)
        e = np.array([0, 2,
                      0, 3,
                      1, 4,
                      1, 5,
                      2, 4,
                      2, 6,
                      3, 5,
                      3, 6,
                      4, 7,
                      5, 7,
                      6, 7]) # see the picture in init
        for i in range(11):
            self.edges = np.hstack((self.edges, np.sort(np.vstack((self.t[e[2*i], :],
                                                                   self.t[e[2*i+1], :])), axis=0)))

        # unique edges
        self.edges, ixa, ixb = np.unique(self.edges, axis=1, return_index=True, return_inverse=True)
        self.edges = np.ascontiguousarray(self.edges)

        self.t2e = ixb.reshape((12, self.t.shape[1]))

        # define facets
        self.facets = np.vstack((self.t[0, :],
                                 self.t[1, :],
                                 self.t[4, :],
                                 self.t[2, :]))
        f = np.array([0, 2, 6, 3,
                      0, 3, 5, 1,
                      2, 4, 7, 6,
                      1, 5, 7, 4,
                      3, 6, 7, 5])
        for i in range(5):
            self.facets = np.hstack((self.facets, np.vstack((self.t[f[4*i], :],
                                                             self.t[f[4*i+1], :],
                                                             self.t[f[4*i+2], :],
                                                             self.t[f[4*i+3], :]))))

        sorted_facets = np.sort(self.facets, axis=0)

        # unique facets
        sorted_facets, ixa, ixb = np.unique(sorted_facets, axis=1, return_index=True, return_inverse=True)
        self.facets = np.ascontiguousarray(self.facets[:, ixa])

        self.t2f = ixb.reshape((6, self.t.shape[1]))

        # build facet-to-hexa mapping: 2 (hexes) x Nfacets
        e_tmp = np.hstack((self.t2f[0, :], self.t2f[1, :],
                           self.t2f[2, :], self.t2f[3, :],
                           self.t2f[4, :], self.t2f[5, :]))
        t_tmp = np.tile(np.arange(self.t.shape[1]), (1, 6))[0]

        e_first, ix_first = np.unique(e_tmp, return_index=True)
        e_last, ix_last = np.unique(e_tmp[::-1], return_index=True)
        ix_last = e_tmp.shape[0] - ix_last - 1

        self.f2t = np.zeros((2, self.facets.shape[1]), dtype=np.int64)
        self.f2t[0, e_first] = t_tmp[ix_first]
        self.f2t[1, e_last] = t_tmp[ix_last]

        ## second row to zero if repeated (i.e., on boundary)
        self.f2t[1, np.nonzero(self.f2t[0, :] == self.f2t[1, :])[0]] = -1

    def _uniform_refine(self):
        """Perform a single mesh refine that halves 'h'. Each hex is
        split into 8."""
        # rename variables
        t = self.t
        p = self.p
        e = self.edges
        f = self.facets
        sz = p.shape[1]
        t2e = self.t2e + sz
        t2f = self.t2f + np.max(t2e) + 1
        # hex middle point
        mid = range(self.t.shape[1]) + np.max(t2f) + 1
        # new vertices are the midpoints of edges ...
        newp1 = 0.5*np.vstack((p[0, e[0, :]] + p[0, e[1, :]],
                               p[1, e[0, :]] + p[1, e[1, :]],
                               p[2, e[0, :]] + p[2, e[1, :]]))
        # ... midpoints of facets ...
        newp2 = 0.25*np.vstack((p[0, f[0, :]] + p[0, f[1, :]] +
                                p[0, f[2, :]] + p[0, f[3, :]],
                                p[1, f[0, :]] + p[1, f[1, :]] +
                                p[1, f[2, :]] + p[1, f[3, :]],
                                p[2, f[0, :]] + p[2, f[1, :]] +
                                p[2, f[2, :]] + p[2, f[3, :]]))
        # ... and element middle points
        newp3 = 0.125*np.vstack((p[0, t[0, :]] + p[0, t[1, :]] +
                                 p[0, t[2, :]] + p[0, t[3, :]] +
                                 p[0, t[4, :]] + p[0, t[5, :]] +
                                 p[0, t[6, :]] + p[0, t[7, :]],
                                 p[1, t[0, :]] + p[1, t[1, :]] +
                                 p[1, t[2, :]] + p[1, t[3, :]] +
                                 p[1, t[4, :]] + p[1, t[5, :]] +
                                 p[1, t[6, :]] + p[1, t[7, :]],
                                 p[2, t[0, :]] + p[2, t[1, :]] +
                                 p[2, t[2, :]] + p[2, t[3, :]] +
                                 p[2, t[4, :]] + p[2, t[5, :]] +
                                 p[2, t[6, :]] + p[2, t[7, :]]))
        newp = np.hstack((p, newp1, newp2, newp3))
        # build new hex indexing (this requires some serious meditation)
        newt = np.vstack((t[0, :],
                          t2e[0, :],
                          t2e[1, :],
                          t2e[2, :],
                          t2f[0, :],
                          t2f[2, :],
                          t2f[1, :],
                          mid))
        newt = np.hstack((newt, np.vstack((t2e[0, :],
                                           t[1, :],
                                           t2f[0, :],
                                           t2f[2, :],
                                           t2e[3, :],
                                           t2e[4, :],
                                           mid,
                                           t2f[4, :]))))
        newt = np.hstack((newt, np.vstack((t2e[1, :],
                                           t2f[0, :],
                                           t[2, :],
                                           t2f[1, :],
                                           t2e[5, :],
                                           mid,
                                           t2e[6, :],
                                           t2f[3, :]
                                           ))))
        newt = np.hstack((newt, np.vstack((t2e[2, :],
                                           t2f[2, :],
                                           t2f[1, :],
                                           t[3, :],
                                           mid,
                                           t2e[7, :],
                                           t2e[8, :],
                                           t2f[5, :]
                                           ))))
        newt = np.hstack((newt, np.vstack((t2f[0, :],
                                           t2e[3, :],
                                           t2e[5, :],
                                           mid,
                                           t[4, :],
                                           t2f[4, :],
                                           t2f[3, :],
                                           t2e[9, :]
                                           ))))
        newt = np.hstack((newt, np.vstack((t2f[2, :],
                                           t2e[4, :],
                                           mid,
                                           t2e[7, :],
                                           t2f[4, :],
                                           t[5, :],
                                           t2f[5, :],
                                           t2e[10, :],
                                           ))))
        newt = np.hstack((newt, np.vstack((t2f[1, :],
                                           mid,
                                           t2e[6, :],
                                           t2e[8, :],
                                           t2f[3, :],
                                           t2f[5, :],
                                           t[6, :],
                                           t2e[11, :]
                                           ))))
        newt = np.hstack((newt, np.vstack((mid,
                                           t2f[4, :],
                                           t2f[3, :],
                                           t2f[5, :],
                                           t2e[9, :],
                                           t2e[10, :],
                                           t2e[11, :],
                                           t[7, :]
                                           ))))
        # update fields
        self.p = newp
        self.t = newt

        self._build_mappings()

        # TODO implement prolongation

    def save(self,
            filename: str,
            pointData: Optional[Union[ndarray, Dict[str, ndarray]]] = None,
            cellData: Optional[Union[ndarray, Dict[str, ndarray]]] = None):
        """Export the mesh and fields using meshio. (Hexahedron version.)

        Parameters
        ----------
        filename
            The filename for vtk-file.
        pointData
            ndarray for one output or dict for multiple
        cellData
            ndarray for one output or dict for multiple

        """
        import meshio

        # vtk requires a different ordering
        t = self.t[[0, 3, 6, 2, 1, 5, 7, 4], :]

        if pointData is not None:
            if type(pointData) != dict:
                pointData = {'0':pointData}

        if cellData is not None:
            if type(cellData) != dict:
                cellData = {'0':cellData}

        cells = { 'hexahedron' : t.T }
        mesh = meshio.Mesh(self.p.T, cells, pointData, cellData)
        meshio.write(filename, mesh)

    def mapping(self):
        return MappingIsoparametric(self, ElementHex1(), ElementQuad1())
