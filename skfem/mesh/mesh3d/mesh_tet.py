import numpy as np
import matplotlib.pyplot as plt

from skfem.mapping import MappingAffine

from ..mesh import MeshType
from .mesh3d import Mesh3D

from typing import Type, Optional, Dict
from numpy import ndarray


class MeshTet(Mesh3D):
    """A mesh consisting of tetrahedral elements.

    The different constructors are:

    - :meth:`~skfem.mesh.MeshTet.__init__`
    - :meth:`~skfem.mesh.MeshTet.load` (requires meshio)
    - :meth:`~skfem.mesh.MeshTet.init_tensor`

    Attributes
    ----------
    facets
        Each column contains a triplet of column indices to MeshTet.p
        (3 x Nfacets).  Order: (0, 1, 2) (0, 1, 3) (0, 2, 3) (1, 2, 3)
    f2t
        Each column contains a pair of column indices to MeshTet.t or
        -1 on the second row if the facet is located on the boundary
        (2 x Nfacets).
    t2f
        Each column contains four indices to MeshTet.facets (4 x Nelements).
    edges
        Each column corresponds to an edge and contains two indices to
        MeshTet.p (2 x Nedges).
        Order: (0, 1) (1, 2) (0, 2) (0, 3) (1, 3) (2, 3)
    t2e
        Each column contains six indices to MeshTet.edges (6 x Nelements).

    """

    refdom: str = "tet"
    brefdom: str = "tri"
    meshio_type: str = "tetra"
    name: str = "Tetrahedral"

    def __init__(self,
                 p: Optional[ndarray] = None,
                 t: Optional[ndarray] = None,
                 boundaries: Optional[Dict[str, ndarray]] = None,
                 subdomains: Optional[Dict[str, ndarray]] = None,
                 validate=True):
        if p is None and t is None:
            p = np.array([[0., 0., 0.],
                          [0., 0., 1.],
                          [0., 1., 0.],
                          [1., 0., 0.],
                          [0., 1., 1.],
                          [1., 0., 1.],
                          [1., 1., 0.],
                          [1., 1., 1.]]).T
            t = np.array([[0, 1, 2, 3],
                          [3, 5, 1, 7],
                          [2, 3, 6, 7],
                          [2, 3, 1, 7],
                          [1, 2, 4, 7]]).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        self.boundaries = boundaries
        self.subdomains = subdomains
        super(MeshTet, self).__init__()
        if validate:
            self._validate()
        self.enable_facets = True
        self._build_mappings()

    @classmethod
    def init_tensor(cls: Type[MeshType],
                    x: ndarray,
                    y: ndarray,
                    z: ndarray) -> MeshType:
        """Initialise a tensor product mesh.

        Parameters
        ----------
        x
            The nodal coordinates in dimension x
        y
            The nodal coordinates in dimension y
        z
            The nodal coordinates in dimension z

        License
        -------

        From: https://github.com/nschloe/meshzoo

        Copyright (c) 2016-2018 Nico Schlömer

        Permission is hereby granted, free of charge, to any person obtaining a
        copy of this software and associated documentation files (the
        "Software"), to deal in the Software without restriction, including
        without limitation the rights to use, copy, modify, merge, publish,
        distribute, sublicense, and/or sell copies of the Software, and to
        permit persons to whom the Software is furnished to do so, subject to
        the following conditions:

        The above copyright notice and this permission notice shall be included
        in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
        OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
        NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
        LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
        OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
        WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

        """
        # Create the vertices.
        nx, ny, nz = len(x), len(y), len(z)
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        p = np.array([x, y, z]).T.reshape(-1, 3).T

        # Create the elements.
        a0 = np.add.outer(np.array(range(nx - 1)), nx*np.array(range(ny - 1)))
        a = np.add.outer(a0, nx*ny*np.array(range(nz - 1)))

        elems0 = np.concatenate([a[..., None],
                                 a[..., None] + nx,
                                 a[..., None] + 1,
                                 a[..., None] + nx*ny], axis=3)

        elems0[1::2, 0::2, 0::2, 0] += 1
        elems0[0::2, 1::2, 0::2, 0] += 1
        elems0[0::2, 0::2, 1::2, 0] += 1
        elems0[1::2, 1::2, 1::2, 0] += 1

        elems0[1::2, 0::2, 0::2, 1] += 1
        elems0[0::2, 1::2, 0::2, 1] += 1
        elems0[0::2, 0::2, 1::2, 1] += 1
        elems0[1::2, 1::2, 1::2, 1] += 1

        elems0[1::2, 0::2, 0::2, 2] -= 1
        elems0[0::2, 1::2, 0::2, 2] -= 1
        elems0[0::2, 0::2, 1::2, 2] -= 1
        elems0[1::2, 1::2, 1::2, 2] -= 1

        elems0[1::2, 0::2, 0::2, 3] += 1
        elems0[0::2, 1::2, 0::2, 3] += 1
        elems0[0::2, 0::2, 1::2, 3] += 1
        elems0[1::2, 1::2, 1::2, 3] += 1

        elems1 = np.concatenate([a[..., None] + nx,
                                 a[..., None] + 1 + nx,
                                 a[..., None] + 1,
                                 a[..., None] + 1 + nx + nx*ny], axis=3)

        elems1[1::2, 0::2, 0::2, 0] += 1
        elems1[0::2, 1::2, 0::2, 0] += 1
        elems1[0::2, 0::2, 1::2, 0] += 1
        elems1[1::2, 1::2, 1::2, 0] += 1

        elems1[1::2, 0::2, 0::2, 1] -= 1
        elems1[0::2, 1::2, 0::2, 1] -= 1
        elems1[0::2, 0::2, 1::2, 1] -= 1
        elems1[1::2, 1::2, 1::2, 1] -= 1

        elems1[1::2, 0::2, 0::2, 2] -= 1
        elems1[0::2, 1::2, 0::2, 2] -= 1
        elems1[0::2, 0::2, 1::2, 2] -= 1
        elems1[1::2, 1::2, 1::2, 2] -= 1

        elems1[1::2, 0::2, 0::2, 3] -= 1
        elems1[0::2, 1::2, 0::2, 3] -= 1
        elems1[0::2, 0::2, 1::2, 3] -= 1
        elems1[1::2, 1::2, 1::2, 3] -= 1

        elems2 = np.concatenate([a[..., None] + nx,
                                 a[..., None] + 1,
                                 a[..., None] + nx*ny,
                                 a[..., None] + 1 + nx + nx*ny], axis=3)

        elems2[1::2, 0::2, 0::2, 0] += 1
        elems2[0::2, 1::2, 0::2, 0] += 1
        elems2[0::2, 0::2, 1::2, 0] += 1
        elems2[1::2, 1::2, 1::2, 0] += 1

        elems2[1::2, 0::2, 0::2, 1] -= 1
        elems2[0::2, 1::2, 0::2, 1] -= 1
        elems2[0::2, 0::2, 1::2, 1] -= 1
        elems2[1::2, 1::2, 1::2, 1] -= 1

        elems2[1::2, 0::2, 0::2, 2] += 1
        elems2[0::2, 1::2, 0::2, 2] += 1
        elems2[0::2, 0::2, 1::2, 2] += 1
        elems2[1::2, 1::2, 1::2, 2] += 1

        elems2[1::2, 0::2, 0::2, 3] -= 1
        elems2[0::2, 1::2, 0::2, 3] -= 1
        elems2[0::2, 0::2, 1::2, 3] -= 1
        elems2[1::2, 1::2, 1::2, 3] -= 1

        elems3 = np.concatenate([a[..., None] + nx,
                                 a[..., None] + nx*ny,
                                 a[..., None] + nx + nx*ny,
                                 a[..., None] + 1 + nx + nx*ny], axis=3)

        elems3[1::2, 0::2, 0::2, 0] += 1
        elems3[0::2, 1::2, 0::2, 0] += 1
        elems3[0::2, 0::2, 1::2, 0] += 1
        elems3[1::2, 1::2, 1::2, 0] += 1

        elems3[1::2, 0::2, 0::2, 1] += 1
        elems3[0::2, 1::2, 0::2, 1] += 1
        elems3[0::2, 0::2, 1::2, 1] += 1
        elems3[1::2, 1::2, 1::2, 1] += 1

        elems3[1::2, 0::2, 0::2, 2] += 1
        elems3[0::2, 1::2, 0::2, 2] += 1
        elems3[0::2, 0::2, 1::2, 2] += 1
        elems3[1::2, 1::2, 1::2, 2] += 1

        elems3[1::2, 0::2, 0::2, 3] -= 1
        elems3[0::2, 1::2, 0::2, 3] -= 1
        elems3[0::2, 0::2, 1::2, 3] -= 1
        elems3[1::2, 1::2, 1::2, 3] -= 1

        elems4 = np.concatenate([a[..., None] + 1,
                                 a[..., None] + nx*ny,
                                 a[..., None] + 1 + nx + nx*ny,
                                 a[..., None] + 1 + nx*ny], axis=3)

        elems4[1::2, 0::2, 0::2, 0] -= 1
        elems4[0::2, 1::2, 0::2, 0] -= 1
        elems4[0::2, 0::2, 1::2, 0] -= 1
        elems4[1::2, 1::2, 1::2, 0] -= 1

        elems4[1::2, 0::2, 0::2, 1] += 1
        elems4[0::2, 1::2, 0::2, 1] += 1
        elems4[0::2, 0::2, 1::2, 1] += 1
        elems4[1::2, 1::2, 1::2, 1] += 1

        elems4[1::2, 0::2, 0::2, 2] -= 1
        elems4[0::2, 1::2, 0::2, 2] -= 1
        elems4[0::2, 0::2, 1::2, 2] -= 1
        elems4[1::2, 1::2, 1::2, 2] -= 1

        elems4[1::2, 0::2, 0::2, 3] -= 1
        elems4[0::2, 1::2, 0::2, 3] -= 1
        elems4[0::2, 0::2, 1::2, 3] -= 1
        elems4[1::2, 1::2, 1::2, 3] -= 1

        t = np.vstack([elems0.reshape(-1, 4),
                       elems1.reshape(-1, 4),
                       elems2.reshape(-1, 4),
                       elems3.reshape(-1, 4),
                       elems4.reshape(-1, 4)]).T

        p = np.ascontiguousarray(p)
        t = np.ascontiguousarray(t)

        return cls(p, t)

    def _build_mappings(self):
        """Build element-to-facet, element-to-edges, etc. mappings."""
        # define edges: in the order (0,1) (1,2) (0,2) (0,3) (1,3) (2,3)
        self.edges = np.sort(np.vstack((self.t[0, :], self.t[1, :])), axis=0)
        e = np.array([1, 2,
                      0, 2,
                      0, 3,
                      1, 3,
                      2, 3])
        for i in range(5):
            self.edges = np.hstack((
                self.edges,
                np.sort(np.vstack((self.t[e[2*i], :],
                                   self.t[e[2*i+1], :])), axis=0)
            ))

        # unique edges
        self.edges, ixa, ixb = np.unique(self.edges,
                                         axis=1,
                                         return_index=True,
                                         return_inverse=True)
        self.edges = np.ascontiguousarray(self.edges)

        self.t2e = ixb.reshape((6, self.t.shape[1]))

        # define facets
        if self.enable_facets:
            self.facets = np.sort(np.vstack((self.t[0, :],
                                             self.t[1, :],
                                             self.t[2, :])), axis=0)
            f = np.array([0, 1, 3,
                          0, 2, 3,
                          1, 2, 3])
            for i in range(3):
                self.facets = np.hstack((
                    self.facets,
                    np.sort(np.vstack((self.t[f[2*i], :],
                                       self.t[f[2*i+1], :],
                                       self.t[f[2*i+2]])), axis=0)
                ))

            # unique facets
            self.facets, ixa, ixb = np.unique(self.facets,
                                              axis=1,
                                              return_index=True,
                                              return_inverse=True)
            self.facets = np.ascontiguousarray(self.facets)

            self.t2f = ixb.reshape((4, self.t.shape[1]))

            # build facet-to-tetra mapping: 2 (tets) x Nfacets
            e_tmp = np.hstack((self.t2f[0, :], self.t2f[1, :],
                               self.t2f[2, :], self.t2f[3, :]))
            t_tmp = np.tile(np.arange(self.t.shape[1]), (1, 4))[0]

            e_first, ix_first = np.unique(e_tmp, return_index=True)
            # this emulates matlab unique(e_tmp,'last')
            e_last, ix_last = np.unique(e_tmp[::-1], return_index=True)
            ix_last = e_tmp.shape[0] - ix_last-1

            self.f2t = np.zeros((2, self.facets.shape[1]), dtype=np.int64)
            self.f2t[0, e_first] = t_tmp[ix_first]
            self.f2t[1, e_last] = t_tmp[ix_last]

            # second row to zero if repeated (i.e., on boundary)
            self.f2t[1, np.nonzero(self.f2t[0, :] == self.f2t[1, :])[0]] = -1

    def refine(self, N=None):
        """Refine the mesh, tetrahedral optimization.
        
        Parameters
        ----------
        N : (optional) int
            Perform N refinements.

        """
        if N is None:
            return self._uniform_refine()
        else:
            self.enable_facets = False
            for itr in range(N-1):
                self._uniform_refine()
            self.enable_facets = True
            self._uniform_refine()

    def _uniform_refine(self):
        """Perform a single mesh refine.

        Let the nodes of a tetrahedron be numbered as 0, 1, 2 and 3.
        It is assumed that the edges in self.t2e are given in the order

          I=(0,1), II=(1,2), III=(0,2), IV=(0,3), V=(1,3), VI=(2,3)

        by self._build_mappings(). Let I denote the midpoint of the edge
        (0,1), II denote the midpoint of the edge (1,2), etc. Then each
        tetrahedron is split into eight smaller subtetrahedra as follows.

        The first four subtetrahedra have the following nodes:

          1. (0,I,III,IV)
          2. (1,I,II,V)
          3. (2,II,III,VI)
          4. (3,IV,V,VI)

        The remaining middle-portion of the original tetrahedron consists
        of a union of two mirrored pyramids. This bi-pyramid can be splitted
        into four tetrahedra in a three different ways by connecting the
        midpoints of two opposing edges (there are three different pairs
        of opposite edges).

        For each tetrahedra in the original mesh, we split the bi-pyramid
        in such a way that the connection between the opposite edges
        is shortest. This minimizes the shape-regularity constant of
        the resulting mesh family.

        """
        # rename variables
        t = self.t
        p = self.p
        e = self.edges
        sz = p.shape[1]
        t2e = self.t2e + sz
        # new vertices are the midpoints of edges
        newp = 0.5*np.vstack((p[0, e[0, :]] + p[0, e[1, :]],
                              p[1, e[0, :]] + p[1, e[1, :]],
                              p[2, e[0, :]] + p[2, e[1, :]]))
        newp = np.hstack((p, newp))
        # new tets
        newt = np.vstack((t[0, :], t2e[0, :], t2e[2, :], t2e[3, :]))
        newt = np.hstack((newt, np.vstack((t[1, :], t2e[0, :], t2e[1, :], t2e[4, :]))))
        newt = np.hstack((newt, np.vstack((t[2, :], t2e[1, :], t2e[2, :], t2e[5, :]))))
        newt = np.hstack((newt, np.vstack((t[3, :], t2e[3, :], t2e[4, :], t2e[5, :]))))
        # compute middle pyramid diagonal lengths and choose shortest
        d1 = ((newp[0, t2e[2, :]] - newp[0, t2e[4, :]])**2 +
              (newp[1, t2e[2, :]] - newp[1, t2e[4, :]])**2)
        d2 = ((newp[0, t2e[1, :]] - newp[0, t2e[3, :]])**2 +
              (newp[1, t2e[1, :]] - newp[1, t2e[3, :]])**2)
        d3 = ((newp[0, t2e[0, :]] - newp[0, t2e[5, :]])**2 +
              (newp[1, t2e[0, :]] - newp[1, t2e[5, :]])**2)
        I1 = d1 < d2
        I2 = d1 < d3
        I3 = d2 < d3
        c1 = I1*I2
        c2 = (~I1)*I3
        c3 = (~I2)*(~I3)
        # splitting the pyramid in the middle.
        # diagonals are [2,4], [1,3] and [0,5]
        # CASE 1: diagonal [2,4]
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[0, c1], t2e[1, c1]))))
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[0, c1], t2e[3, c1]))))
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[1, c1], t2e[5, c1]))))
        newt = np.hstack((newt, np.vstack((t2e[2, c1], t2e[4, c1],
                                           t2e[3, c1], t2e[5, c1]))))
        # CASE 2: diagonal [1,3]
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[0, c2], t2e[4, c2]))))
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[4, c2], t2e[5, c2]))))
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[5, c2], t2e[2, c2]))))
        newt = np.hstack((newt, np.vstack((t2e[1, c2], t2e[3, c2],
                                           t2e[2, c2], t2e[0, c2]))))
        # CASE 3: diagonal [0,5]
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[1, c3], t2e[4, c3]))))
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[4, c3], t2e[3, c3]))))
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[3, c3], t2e[2, c3]))))
        newt = np.hstack((newt, np.vstack((t2e[0, c3], t2e[5, c3],
                                           t2e[2, c3], t2e[1, c3]))))
        # update fields
        self.p = newp
        self.t = newt

        self._build_mappings()

    def draw(self):
        """Draw the (surface) mesh."""
        from mpl_toolkits.mplot3d import Axes3D
        
        bnd_facets = self.boundary_facets()
        fig = plt.figure()
        ax = Axes3D(fig)
        indexing = self.facets[:, bnd_facets].T

        ax.plot_trisurf(self.p[0, :], self.p[1, :], self.p[2,:],
                        triangles=indexing, cmap=plt.cm.viridis, edgecolor='k')
        ax.set_axis_off()
        return ax

    def shapereg(self):
        """Return the largest shape-regularity constant."""
        def edgelen(n):
            return np.sqrt(np.sum((self.p[:, self.edges[0, self.t2e[n, :]]] -
                                   self.p[:, self.edges[1, self.t2e[n, :]]])**2,
                                  axis=0))
        edgelenmat = np.vstack(tuple(edgelen(i) for i in range(6)))
        return np.max(np.max(edgelenmat, axis=0)/np.min(edgelenmat, axis=0))

    def mapping(self):
        return MappingAffine(self)
