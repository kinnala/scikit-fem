# -*- coding: utf-8 -*-
"""
Mesh module contains different types of finite element meshes.

Currently implemented mesh types are

    * :class:`skfem.mesh.MeshTri`, a triangular mesh
    * :class:`skfem.mesh.MeshTet`, a tetrahedral mesh
    * :class:`skfem.mesh.MeshQuad`, a mesh consisting of quadrilaterals
    * :class:`skfem.mesh.MeshLine`, one-dimensional mesh
    * :class:`skfem.mesh.MeshLineMortar`, interface mesh between two 2D meshes

Examples
--------

Obtain a three times refined mesh of the unit square and draw it.

.. code-block:: python

    from skfem.mesh import MeshTri
    m = MeshTri()
    m.refine(3)
    m.draw()
    m.show()

"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import warnings
from scipy.sparse import coo_matrix

import skfem.mapping


class Mesh(object):
    """Finite element mesh."""

    refdom = "none"  #: A string defining type of the mesh
    brefdom = "none"  #: A string defining type of the boundary mesh

    p = np.array([])  #: The vertices of the mesh, size: dim x Npoints
    t = np.array([])  #: The element connectivity, size: verts/elem x Nelems

    def __init__(self):
        if self.p is not None:
            if self.p.flags['F_CONTIGUOUS']:
                if self.p.shape[1]>1000:
                    warnings.warn("Mesh.__init__(): Transforming " +
                            "over 100 vertices to C_CONTIGUOUS.")
                self.p = np.ascontiguousarray(self.p)
        if self.t is not None:
            if self.t.flags['F_CONTIGUOUS']:
                if self.t.shape[1]>1000:
                    warnings.warn("Mesh.__init__(): Transforming " +
                            "over 100 elements to C_CONTIGUOUS.")
                self.t = np.ascontiguousarray(self.t)

    def show(self):
        """Call the correct pyplot show commands after plotting."""
        plt.show()

    def dim(self):
        """Return the spatial dimension of the mesh."""
        return int(self.p.shape[0])

    def mapping(self):
        """Default local-to-global mapping for the mesh."""
        raise NotImplementedError("Default mapping not implemented!")

    def _uniform_refine(self):
        """Perform a single uniform mesh refinement."""
        raise NotImplementedError("Single refine not implemented for this mesh type!")

    def refine(self, N=None):
        """Refine the mesh."""
        if N is None:
            return self._uniform_refine()
        else:
            for itr in range(N):
                self._uniform_refine()

    def remove_elements(self, element_indices):
        """Construct new mesh with elements removed
        based on their indices.

        Parameters
        ----------
        element_indices : numpy array
            List of element indices to remove.

        Returns
        -------
        skfem.Mesh
            A new mesh object with elements removed as per requested.
        """
        keep = np.setdiff1d(np.arange(self.t.shape[1]), element_indices)
        newt = self.t[:, keep]
        ptix = np.unique(newt)
        reverse = np.zeros(self.p.shape[1])
        reverse[ptix] = np.arange(len(ptix))
        newt = reverse[newt]
        newp = self.p[:, ptix]
        return newp, newt.astype(np.intp)

    def scale(self, scale):
        """Scale the mesh.

        Parameters
        ----------
        scale : float OR tuple of size dim
            Scale each dimension by a factor. If a floating
            point number is provided, same scale is used
            for each dimension.
        """
        for itr in range(int(self.dim())):
            if isinstance(scale, tuple):
                self.p[itr, :] *= scale[itr]
            else:
                self.p[itr, :] *= scale

    def translate(self, vec):
        """Translate the mesh.

        Parameters
        ----------
        vec : tuple of size dim
            Translate the mesh by a vector.
        """
        for itr in range(int(self.dim())):
            self.p[itr, :] += vec[itr]

    def _validate(self):
        """Perform mesh validity checks."""
        # check that element connectivity contains integers
        # NOTE: this is neccessary for some plotting functionality
        if not np.issubdtype(self.t[0, 0], int):
            msg = ("Mesh._validate(): Element connectivity "
                   "must consist of integers.")
            raise Exception(msg)
        # check that vertex matrix has "correct" size
        if self.p.shape[0] > 3:
            msg = ("Mesh._validate(): We do not allow meshes "
                   "embedded into larger than 3-dimensional "
                   "Euclidean space! Please check that "
                   "the given vertex matrix is of size Ndim x Nvertices.")
            raise Exception(msg)
        # check that element connectivity matrix has correct size
        nvertices = {'line': 2, 'tri': 3, 'quad': 4, 'tet': 4}
        if self.t.shape[0] != nvertices[self.refdom]:
            msg = ("Mesh._validate(): The given connectivity "
                   "matrix has wrong shape!")
            raise Exception(msg)
        # check that there are no duplicate points
        tmp = np.ascontiguousarray(self.p.T)
        if self.p.shape[1] != np.unique(tmp.view([('', tmp.dtype)]
                                                 * tmp.shape[1])).shape[0]:
            msg = "Mesh._validate(): Mesh contains duplicate vertices."
            warnings.warn(msg)
        # check that all points are at least in some element
        if len(np.setdiff1d(np.arange(self.p.shape[1]), np.unique(self.t))):
            msg = ("Mesh._validate(): Mesh contains a vertex "
                   "not belonging to any element.")
            raise Exception(msg)

    def jiggle(self, z=0.2):
        """Jiggle the interior nodes of the mesh.

        Parameters
        ----------
        z : (OPTIONAL, default=0.2) float
            Mesh parameter is multiplied by this number. The resulting number
            corresponds to the standard deviation of the jiggle.
        """
        y = z*self.param()
        I = self.interior_nodes()
        self.p[0, I] = self.p[0, I] + y*np.random.rand(len(I))
        self.p[1, I] = self.p[1, I] + y*np.random.rand(len(I))


class Mesh2D(Mesh):
    """Two dimensional meshes, common methods."""

    def boundary_nodes(self):
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:, self.boundary_facets()])

    def nodes_satisfying(self, test):
        """Return nodes that satisfy some condition."""
        return np.nonzero(test(self.p[0, :], self.p[1, :]))[0]

    def draw_nodes(self, nodes, mark='bo'):
        """Highlight some nodes."""
        plt.plot(self.p[0, nodes], self.p[1, nodes], mark)

    def param(self):
        """Return mesh parameter."""
        return np.max(np.sqrt(np.sum((self.p[:, self.facets[0, :]] -
                                      self.p[:, self.facets[1, :]])**2, axis=0)))

    def interior_nodes(self):
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0, self.p.shape[1]), self.boundary_nodes())

    def facets_satisfying(self, test):
        """Return facets whose midpoints satisfy some condition."""
        mx = 0.5*(self.p[0, self.facets[0, :]] + self.p[0, self.facets[1, :]])
        my = 0.5*(self.p[1, self.facets[0, :]] + self.p[1, self.facets[1, :]])
        return np.nonzero(test(mx, my))[0]

    def elements_satisfying(self, test):
        """Return elements whose midpoints satisfy some condition."""
        mx = np.sum(self.p[0, self.t], axis=0)/self.t.shape[0]
        my = np.sum(self.p[1, self.t], axis=0)/self.t.shape[0]
        return np.nonzero(test(mx, my))[0]

    def interior_facets(self):
        """Return an array of interior facet indices."""
        return np.nonzero(self.f2t[1, :] >= 0)[0]

    def boundary_facets(self):
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1, :] == -1)[0]

    def draw(self, ax=None, numbering=False):
        """Draw the mesh."""
        if ax is None:
            # create new figure
            fig = plt.figure()
            ax = fig.add_subplot(111)
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
        ax.plot(xs, ys, 'k')
        if numbering:
            mx = np.sum(self.p[0, self.t], axis=0)/self.t.shape[0]
            my = np.sum(self.p[1, self.t], axis=0)/self.t.shape[0]
            for itr in range(self.t.shape[1]):
                ax.text(mx[itr], my[itr], str(itr))
        return ax

    def mirror_mesh(self, a, b, c):
        """Mirror a mesh by a line."""
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


class MeshLineMortar(Mesh):
    """One-dimensional mortar mesh at the interface of two 2D meshes."""

    refdom = "line"
    brefdom = "point"

    def __init__(self, mesh1, mesh2, rule, param, debug_plot=False):
        """
        Rule should be so that its positive only if
        point belongs to the subset and increases
        so that different points can be distinquished.
        """
        p1_ix = mesh1.nodes_satisfying(rule)
        p2_ix = mesh2.nodes_satisfying(rule)

        p1 = mesh1.p[:, p1_ix]
        p2 = mesh2.p[:, p2_ix]
        _, ix = np.unique(np.concatenate((param(p1[0, :], p1[1, :]), param(p2[0, :], p2[1, :]))), return_index=True)

        self.p = np.hstack((p1, p2))[:, ix]
        self.facets = np.array([np.arange(np.max(self.p.shape)-1), np.arange(np.max(self.p.shape)-1)+1])

        if debug_plot:
            ax = mesh1.draw()
            mesh2.draw(ax=ax)
            mesh2.draw_nodes(p2_ix, 'rs')
            mesh1.draw_nodes(p1_ix, 'bx')

        # mappings from facets to the original triangles
        # TODO vectorize
        self.f2t = self.facets*0-1
        for itr in range(self.facets.shape[1]):
            mx = .5*(self.p[0, self.facets[0, itr]] + self.p[0, self.facets[1, itr]])
            my = .5*(self.p[1, self.facets[0, itr]] + self.p[1, self.facets[1, itr]])
            val = param(mx, my)
            for jtr in mesh1.boundary_facets():  
                fix1 = mesh1.facets[0, jtr]
                x1 = mesh1.p[0, fix1]
                y1 = mesh1.p[1, fix1]
                fix2 = mesh1.facets[1, jtr]
                x2 = mesh1.p[0, fix2]
                y2 = mesh1.p[1, fix2]
                if rule(x1, y1) > 0 or rule(x2, y2) > 0:
                    if val > param(x1, y1) and val < param(x2, y2):
                        # OK
                        self.f2t[0, itr] = jtr
                        break
                    elif val < param(x1, y1) and val > param(x2, y2):
                        # OK
                        self.f2t[0, itr] = jtr
                        break
            for jtr in mesh2.boundary_facets():  
                fix1 = mesh2.facets[0, jtr]
                x1 = mesh2.p[0, fix1]
                y1 = mesh2.p[1, fix1]
                fix2 = mesh2.facets[1, jtr]
                x2 = mesh2.p[0, fix2]
                y2 = mesh2.p[1, fix2]
                if rule(x1, y1) > 0 or rule(x2, y2) > 0:
                    if val > param(x1, y1) and val < param(x2, y2):
                        # OK
                        self.f2t[1, itr] = jtr
                        break
                    elif val < param(x1, y1) and val > param(x2, y2):
                        # OK
                        self.f2t[1, itr] = jtr
                        break
        if (self.f2t>-1).all():
            return
        else:
            raise Exception("All mesh facets corresponding to mortar facets not found!")

    def draw(self, color='ro-'):
        """Draw the mesh"""
        xs = []
        ys = []
        for y1, y2, s, t in zip(self.p[1, self.facets[0, :]],
                                self.p[1, self.facets[1, :]],
                                self.p[0, self.facets[0, :]],
                                self.p[0, self.facets[1, :]]):
            xs.append(s)
            xs.append(t)
            xs.append(None)
            ys.append(y1)
            ys.append(y2)
            ys.append(None)
        plt.plot(xs, ys, color)

    def mapping(self):
        return skfem.mapping.MappingAffineMortar(self)

class MeshLine(Mesh):
    """One-dimensional mesh."""

    refdom = "line"
    brefdom = "point"

    def __init__(self, p=None, t=None, validate=True):
        if p is None and t is None:
            p = np.array([[0, 1]])
            t = np.array([[0], [1]])
        elif p is not None and t is None:
            t = np.array([np.arange(np.max(p.shape)-1), np.arange(np.max(p.shape)-1)+1])
        if len(p.shape)==1:
            p = np.array([p]) 
        self.p = p
        self.t = t
        if validate:
            self._validate()
        super(MeshLine, self).__init__()

    def _uniform_refine(self):
        """Perform a single mesh refine that halves 'h'."""
        # rename variables
        t = self.t
        p = self.p

        mid = range(self.t.shape[1]) + np.max(t) + 1
        # new vertices and elements
        newp = np.hstack((p, 0.5*(p[:, self.t[0, :]] + p[:, self.t[1, :]])))
        newt = np.vstack((t[0, :], mid))
        newt = np.hstack((newt, np.vstack((mid, t[1, :]))))
        # update fields
        self.p = newp
        self.t = newt

        # TODO implement prolongation

    def boundary_nodes(self):
        """Find the boundary nodes of the mesh."""
        _, counts = np.unique(self.t.flatten(), return_counts=True)
        return np.nonzero(counts == 1)[0]

    def interior_nodes(self):
        """Find the interior nodes of the mesh."""
        _, counts = np.unique(self.t.flatten(), return_counts=True)
        return np.nonzero(counts == 2)[0]

    def plot(self, u, color='ko-'):
        """Plot a function defined on the nodes of the mesh."""
        xs = []
        ys = []
        for y1, y2, s, t in zip(u[self.t[0, :]],
                                u[self.t[1, :]],
                                self.p[0, self.t[0, :]],
                                self.p[0, self.t[1, :]]):
            xs.append(s)
            xs.append(t)
            xs.append(None)
            ys.append(y1)
            ys.append(y2)
            ys.append(None)
        plt.plot(xs, ys, color)

    def __mul__(self, other):
        """Tensor product mesh."""
        npx = self.p.shape[1]
        npy = other.p.shape[1]
        X, Y = np.meshgrid(np.sort(self.p[0, :]), np.sort(other.p[0, :]))   
        p = np.vstack((X.flatten('F'), Y.flatten('F')))
        ix = np.arange(npx*npy)
        ne = (npx-1)*(npy-1)
        t = np.zeros((4, ne))
        ix = ix.reshape(npy, npx, order='F').copy()
        t[0, :] = ix[0:(npy-1), 0:(npx-1)].reshape(ne, 1, order='F').copy().flatten()
        t[1, :] = ix[1:npy, 0:(npx-1)].reshape(ne, 1, order='F').copy().flatten()
        t[2, :] = ix[1:npy, 1:npx].reshape(ne, 1, order='F').copy().flatten()
        t[3, :] = ix[0:(npy-1), 1:npx].reshape(ne, 1, order='F').copy().flatten()
        return MeshQuad(p, t.astype(np.int64))

    def mapping(self):
        return skfem.mapping.MappingAffine(self)


class MeshQuad(Mesh2D):
    """A mesh consisting of quadrilateral elements."""

    refdom = "quad"
    brefdom = "line"

    def __init__(self, p=None, t=None, initmesh=None, validate=True):
        if p is None and t is None:
            if initmesh is None:
                p = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
                t = np.array([[0, 1, 2, 3]]).T
            elif initmesh is 'refdom':
                p = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]).T
                t = np.array([[0, 1, 2, 3]]).T
            else:
                raise Exception("invalid initmesh keyword.")
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        if validate:
            self._validate()
        self._build_mappings()
        super(MeshQuad, self).__init__()

    def _build_mappings(self):
        # do not sort since order defines counterclockwise order
        # self.t=np.sort(self.t,axis=0)

        # define facets: in the order (0,1) (1,2) (2,3) (0,3)
        self.facets = np.sort(np.vstack((self.t[0, :], self.t[1, :])), axis=0)
        self.facets = np.hstack((self.facets,
                                 np.sort(np.vstack((self.t[1, :],
                                                    self.t[2, :])), axis=0)))
        self.facets = np.hstack((self.facets,
                                 np.sort(np.vstack((self.t[2, :],
                                                    self.t[3, :])), axis=0)))
        self.facets = np.hstack((self.facets,
                                 np.sort(np.vstack((self.t[0, :],
                                                    self.t[3, :])), axis=0)))

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
        """Perform a single mesh refine that halves 'h'.

        Each quadrilateral is split into four subquads."""
        # rename variables
        t = self.t
        p = self.p
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
        newp = np.hstack((p, newp1, newp2))
        # build new quadrilateral definitions
        newt = np.vstack((t[0, :],
                          t2f[0, :],
                          mid,
                          t2f[3, :]))
        newt = np.hstack((newt, np.vstack((t2f[0, :],
                                           t[1, :],
                                           t2f[1, :],
                                           mid))))
        newt = np.hstack((newt, np.vstack((mid,
                                           t2f[1, :],
                                           t[2, :],
                                           t2f[2, :]))))
        newt = np.hstack((newt, np.vstack((t2f[3, :],
                                           mid,
                                           t2f[2, :],
                                           t[3, :]))))
        # update fields
        self.p = newp
        self.t = newt

        self._build_mappings()

        # TODO implement prolongation

    def _splitquads(self, x):
        """Split each quad into a triangle and return MeshTri."""
        if len(x) == self.t.shape[1]:
            # preserve elemental constant functions
            X = np.concatenate((x, x))
        else:
            X = x
        t = self.t[[0, 1, 3], :]
        t = np.hstack((t, self.t[[1, 2, 3]]))
        return MeshTri(self.p, t, validate=False), X

    def plot(self, z, smooth=False, edgecolors=None, ax=None, zlim=None):
        """Visualize nodal or elemental function (2d).

        The quadrilateral mesh is split into triangular mesh (MeshTri) and
        the respective plotting function for the triangular mesh is used.
        """
        m, z = self._splitquads(z)
        return m.plot(z, smooth, ax=ax, zlim=zlim, edgecolors=edgecolors)

    def plot3(self, z, smooth=False, ax=None):
        """Visualize nodal function (3d i.e. three axes).

        The quadrilateral mesh is split into triangular mesh (MeshTri) and
        the respective plotting function for the triangular mesh is used.
        """
        m, z = self._splitquads(z)
        return m.plot3(z, smooth, ax=ax)

    def mapping(self):
        return skfem.mapping.MappingQ1(self)


class MeshTet(Mesh):
    """Tetrahedral mesh."""

    refdom = "tet"
    brefdom = "tri"

    def __init__(self, p=None, t=None, validate=True):
        if p is None and t is None:
            p = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                          [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).T
            t = np.array([[0, 1, 2, 3], [3, 5, 1, 7], [2, 3, 6, 7],
                          [2, 3, 1, 7], [1, 2, 4, 7]]).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        if validate:
            self._validate()
        self.ENABLE_FACETS = True
        self._build_mappings()
        super(MeshTet, self).__init__()

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
            self.edges = np.hstack((self.edges,
                                    np.sort(np.vstack((self.t[e[2*i], :],
                                                       self.t[e[2*i+1], :])),
                                            axis=0)))

        # unique edges
        tmp = np.ascontiguousarray(self.edges.T)
        _, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        self.edges = self.edges[:, ixa]
        self.edges = np.ascontiguousarray(self.edges)
        self.t2e = ixb.reshape((6, self.t.shape[1]))

        # define facets
        if self.ENABLE_FACETS:
            self.facets = np.sort(np.vstack((self.t[0, :],
                                             self.t[1, :],
                                             self.t[2, :])), axis=0)
            f = np.array([0, 1, 3,
                          0, 2, 3,
                          1, 2, 3])
            for i in range(3):
                self.facets = np.hstack((self.facets,
                                         np.sort(np.vstack((self.t[f[2*i], :],
                                                            self.t[f[2*i+1], :],
                                                            self.t[f[2*i+2]])),
                                                 axis=0)))

            # unique facets
            tmp = np.ascontiguousarray(self.facets.T)
            tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                      return_index=True, return_inverse=True)
            self.facets = self.facets[:, ixa]
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
        """Refine the mesh, tetrahedral optimization."""
        if N is None:
            return self._uniform_refine()
        else:
            self.ENABLE_FACETS = False
            for itr in range(N-1):
                self._uniform_refine()
            self.ENABLE_FACETS = True
            self._uniform_refine()

    def nodes_satisfying(self, test):
        """Return nodes that satisfy some condition."""
        return np.nonzero(test(self.p[0, :], self.p[1, :], self.p[2, :]))[0]

    def facets_satisfying(self, test):
        """Return facets whose midpoints satisfy some condition."""
        mx = 0.3333333*(self.p[0, self.facets[0, :]] +
                        self.p[0, self.facets[1, :]] +
                        self.p[0, self.facets[2, :]])
        my = 0.3333333*(self.p[1, self.facets[0, :]] +
                        self.p[1, self.facets[1, :]] +
                        self.p[1, self.facets[2, :]])
        mz = 0.3333333*(self.p[2, self.facets[0, :]] +
                        self.p[2, self.facets[1, :]] +
                        self.p[2, self.facets[2, :]])
        return np.nonzero(test(mx, my, mz))[0]

    def edges_satisfying(self, test):
        """Return edges whose midpoints satisfy some condition."""
        mx = 0.5*(self.p[0, self.edges[0, :]] + self.p[0, self.edges[1, :]])
        my = 0.5*(self.p[1, self.edges[0, :]] + self.p[1, self.edges[1, :]])
        mz = 0.5*(self.p[2, self.edges[0, :]] + self.p[2, self.edges[1, :]])
        return np.nonzero(test(mx, my, mz))[0]

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

        # TODO implement prolongation matrix

    def draw_vertices(self):
        """Draw all vertices using mplot3d."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.p[0, :], self.p[1, :], self.p[2, :])
        return fig

    def boundary_nodes(self):
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:, self.boundary_facets()])

    def boundary_facets(self):
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1, :] == -1)[0]

    def interior_facets(self):
        """Return an array of interior facet indices."""
        return np.nonzero(self.f2t[1, :] >= 0)[0]

    def boundary_edges(self):
        """Return an array of boundary edge indices."""
        bnodes = self.boundary_nodes()[:, None]
        return np.nonzero(np.sum(self.edges[0, :] == bnodes, axis=0) *
                          np.sum(self.edges[1, :] == bnodes, axis=0))[0]

    def interior_nodes(self):
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0, self.p.shape[1]), self.boundary_nodes())

    def param(self):
        """Return (maximum) mesh parameter."""
        return np.max(np.sqrt(np.sum((self.p[:, self.edges[0, :]] -
                                      self.p[:, self.edges[1, :]])**2, axis=0)))

    def shapereg(self):
        """Return the largest shape-regularity constant."""
        def edgelen(n):
            return np.sqrt(np.sum((self.p[:, self.edges[0, self.t2e[n, :]]] -
                                   self.p[:, self.edges[1, self.t2e[n, :]]])**2,
                                  axis=0))
        edgelenmat = np.vstack(tuple(edgelen(i) for i in range(6)))
        return np.max(np.max(edgelenmat, axis=0)/np.min(edgelenmat, axis=0))

    def export_vtk(self, filename, pointData=None, cellData=None):
        """
        Export the mesh and fields to VTK.
        """
        from pyevtk.hl import unstructuredGridToVTK
        from pyevtk.vtk import VtkTetra

        offset = (np.arange(self.t.shape[1])+1)*4
        ctypes = np.zeros(self.t.shape[1]) + VtkTetra.tid
        unstructuredGridToVTK(filename, self.p[0, :], self.p[1, :], self.p[2, :], connectivity=self.t.flatten('F'),
                              offsets=offset, cell_types=ctypes, cellData=cellData, pointData=pointData)

    def mapping(self):
        return skfem.mapping.MappingAffine(self)


class MeshTri(Mesh2D):
    """Triangular mesh."""

    refdom = "tri"
    brefdom = "line"

    def __init__(self, p=None, t=None, validate=True, initmesh=None, sort_t=True):
        if p is None and t is None:
            if initmesh is 'symmetric':
                p = np.array([[0, 1, 1, 0, 0.5],
                              [0, 0, 1, 1, 0.5]], dtype=np.float_)
                t = np.array([[0, 1, 4],
                              [1, 2, 4],
                              [2, 3, 4],
                              [0, 3, 4]], dtype=np.intp).T
            elif initmesh is 'sqsymmetric':
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
            elif initmesh is 'refdom':
                p = np.array([[0, 1, 0],
                              [0, 0, 1]], dtype=np.float_)
                t = np.array([[0, 1, 2]], dtype=np.intp).T
            else:
                p = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float_)
                t = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp).T
        elif p is None or t is None:
            raise Exception("Must provide p AND t or neither")
        self.p = p
        self.t = t
        if validate:
            self._validate()
        self._build_mappings(sort_t=sort_t)
        super(MeshTri, self).__init__()

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

    def interpolator(self, x):
        """Return a function which interpolates values with P1 basis."""
        triang = mtri.Triangulation(self.p[0, :], self.p[1, :], self.t.T)
        interpf = mtri.LinearTriInterpolator(triang, x)
        # contruct an interpolator handle
        def handle(X, Y):
            return interpf(X, Y).data
        return handle

    def const_interpolator(self, x):
        """Return a function which interpolates values with P0 basis."""
        triang = mtri.Triangulation(self.p[0, :], self.p[1, :], self.t.T)
        finder = triang.get_trifinder()
        # construct an interpolator handle
        def handle(X, Y):
            return x[finder(X, Y)]
        return handle

    def smooth(self, c=1.0):
        """Apply smoothing to interior nodes."""
        from skfem.assembly import AssemblerLocal
        from skfem.element import ElementLocalTriP1
        from skfem.utils import direct

        e = ElementLocalTriP1()
        a = AssemblerLocal(self, e)

        K = a.iasm(lambda du,dv: du[0]*dv[0] + du[1]*dv[1])
        M = a.iasm(lambda u,v: u*v)

        I = self.interior_nodes()
        dx = - k*direct(M, K.dot(self.p[0, :]))
        dy = - k*direct(M, K.dot(self.p[1, :]))

        self.p[0, I] += dx[I]
        self.p[1, I] += dy[I]

    def draw_debug(self):
        """Draw without mesh.facets. For debugging self.draw()."""
        fig = plt.figure()
        plt.hold('on')
        for itr in range(self.t.shape[1]):
            plt.plot(self.p[0,self.t[[0,1],itr]], self.p[1,self.t[[0,1],itr]], 'k-')
            plt.plot(self.p[0,self.t[[1,2],itr]], self.p[1,self.t[[1,2],itr]], 'k-')
            plt.plot(self.p[0,self.t[[0,2],itr]], self.p[1,self.t[[0,2],itr]], 'k-')
        return fig

    def plot(self, z, smooth=False, ax=None, zlim=None, edgecolors=None):
        """Visualize nodal or elemental function (2d)."""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
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

    def plot3(self, z, smooth=False, ax=None):
        """Visualize nodal function (3d i.e. three axes)."""
        from mpl_toolkits.mplot3d import Axes3D
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        if len(z) == self.p.shape[1]:
            # use matplotlib
            ts = mtri.Triangulation(self.p[0, :], self.p[1, :], self.t.T)
            ax.plot_trisurf(self.p[0, :], self.p[1, :], z,
                            triangles=ts.triangles,
                            cmap=plt.cm.Spectral)
        elif len(z) == self.t.shape[1]:
            # one value per element (piecewise const)
            nt = self.t.shape[1]
            newt = np.arange(3*nt, dtype=np.int64).reshape((nt, 3))
            newpx = self.p[0, self.t].flatten(order='F')
            newpy = self.p[1, self.t].flatten(order='F')
            newz = np.vstack((z, z, z)).flatten(order='F')
            ts = mtri.Triangulation(newpx, newpx, newt)
            ax.plot_trisurf(newpx, newpy, newz,
                            triangles=ts.triangles,
                            cmap=plt.cm.Spectral)
        elif len(z) == 3*self.t.shape[1]:
            # three values per element (piecewise linear)
            nt = self.t.shape[1]
            newt = np.arange(3*nt, dtype=np.int64).reshape((nt, 3))
            newpx = self.p[0, self.t].flatten(order='F')
            newpy = self.p[1, self.t].flatten(order='F')
            ts = mtri.Triangulation(newpx, newpx, newt)
            ax.plot_trisurf(newpx, newpy, z,
                            triangles=ts.triangles,
                            cmap=plt.cm.Spectral)
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
        return skfem.mapping.MappingAffine(self)
