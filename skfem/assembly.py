# -*- coding: utf-8 -*-
"""
Assembly module contains classes and functions related to the construction
of finite element matrices.

Examples
--------

Assemble a stiffness matrix K for the Dirichlet problem in a unit cube.

>>> from skfem import *
>>> from skfem.models.poisson import *
>>> m = MeshHex()
>>> m.refine(2)
>>> e = ElementHex1()
>>> map = MappingIsoparametric(m, e)
>>> basis = InteriorBasis(m, e, map, 3) # 3 is the order of integration
>>> K = asm(laplace, basis)
>>> K.shape
(125, 125)
"""

import numpy as np
from scipy.sparse import coo_matrix
from skfem.quadrature import get_quadrature


class GlobalBasis():
    """GlobalBasis (abstract) is a combination of Mesh, Element and Mapping
    (and quadrature points).

    The finite element basis is evaluated at global quadrature points and
    cached inside the object.

    Please see the following implementations:

        * InteriorBasis
        * FacetBasis
    """
    def __init__(self, mesh, elem, mapping, intorder):
        self.mapping = mapping

        self.elem = elem
        self.dofnum = Dofnum(mesh, elem)
        self.Nbfun = self.dofnum.t_dof.shape[0]

        self.intorder = intorder

        self.dim = mesh.p.shape[0]
        self.nt = mesh.t.shape[1]

        self.mesh = mesh

        self.refdom = mesh.refdom
        self.brefdom = mesh.brefdom

    def init_gbasis(self, nvals, nqp, order):
        if order == 0:
            return np.empty((self.Nbfun, nvals, nqp))
        else:
            return np.empty((self.Nbfun,) + order*(self.dim,) + (nvals, nqp))

    def default_parameters(self):
        """This is used by assembler to get default parameters for 'w'"""
        raise NotImplementedError("Default parameters not implemented")

    def interpolate(self, w, derivative=False):
        """
        Interpolate a solution vector to quadrature points.

        Parameters
        ----------
        w : ndarray of size Ndofs
            A solution vector
        derivative : (OPTIONAL, default=False) bool
            Return also the derivative

        Returns
        -------
        ndarray of size Nelems x Nqp
            Interpolated solution vector
        """
        nqp = len(self.W)

        W = np.zeros((self.nelems, nqp))
        for j in range(self.Nbfun):
            jdofs = self.dofnum.t_dof[j, :]
            W += w[jdofs][:, None] \
                 * self.phi[j]
        if derivative:
            dW = np.zeros((self.dim, self.nelems, nqp))
            for j in range(self.Nbfun):
                jdofs = self.dofnum.t_dof[j, :]
                for a in range(self.dim):
                    dW[a, :, :] += w[jdofs][:, None] \
                                   * self.dphi[j][a]
            return W, dW
        return W

    def find_dofs(self, test=None, bc=None, boundary=True, dofrows=None,
                  check_vertices=True, check_facets=True, check_edges=True):
        """Helper function for finding DOF indices for BC's.

        Does not test for element interior DOFs since they are not typically
        included in boundary conditions! Uses dofnum of 'u' variable.

        Parameters
        ----------
        test : (OPTIONAL, default=function returning True) lambda
            An anonymous function with Ndim arguments. If returns other than 0
            when evaluated at the DOF location, the respective DOF is included
            in the return set.
        bc : (OPTIONAL, default=zero function) lambda
            The boundary condition value.
        boundary : (OPTIONAL, default=True) bool
            Check only boundary DOFs.
        dofrows : (OPTIONAL, default=None) np.array
            List of rows that are extracted from the DOF structures.
            For example, if each node/facet/edge contains 3 DOFs (say, in three
            dimensional problems x, y and z displacements) you can give [0, 1]
            to consider only two first DOFs.
        check_vertices : (OPTIONAL, default=True) bool
            Include vertex dofs
        check_facets: (OPTIONAL, default=True) bool
            Include facet dofs
        check_edges: (OPTIONAL, default=True) bool
            Include edge dofs (3D only)

        Returns
        -------
        x : np.array
            Solution vector with the BC's
        I : np.array
            Set of DOF numbers set by the function
        """
        if test is None:
            if self.mesh.dim() == 2:
                test = lambda x, y: 0*x + True
            elif self.mesh.dim() == 3:
                test = lambda x, y, z: 0*x + True

        if bc is None:
            if self.mesh.dim() == 2:
                bc = lambda x, y: 0*x
            elif self.mesh.dim() == 3:
                bc = lambda x, y, z: 0*x

        x = np.zeros(self.dofnum.N)

        dofs = np.zeros(0, dtype=np.int64)
        locs = np.zeros((self.mesh.dim(), 0))

        if check_vertices:
            # handle nodes
            N = self.mesh.nodes_satisfying(test)
            if boundary:
                N = np.intersect1d(N, self.mesh.boundary_nodes())
            if dofrows is None:
                Ndofs = self.dofnum.n_dof[:, N]
            else:
                Ndofs = self.dofnum.n_dof[dofrows][:, N]

            Ndofx = np.tile(self.mesh.p[0, N], (Ndofs.shape[0], 1)).flatten()
            Ndofy = np.tile(self.mesh.p[1, N], (Ndofs.shape[0], 1)).flatten()
            if self.mesh.dim() == 3:
                Ndofz = np.tile(self.mesh.p[2, N], (Ndofs.shape[0], 1)).flatten()
                locs = np.hstack((locs, np.vstack((Ndofx, Ndofy, Ndofz))))
            else:
                locs = np.hstack((locs, np.vstack((Ndofx, Ndofy))))

            dofs = np.hstack((dofs, Ndofs.flatten()))

        if check_facets:
            # handle facets
            F = self.mesh.facets_satisfying(test)
            if boundary:
                F = np.intersect1d(F, self.mesh.boundary_facets())
            if dofrows is None:
                Fdofs = self.dofnum.f_dof[:, F]
            else:
                Fdofs = self.dofnum.f_dof[dofrows][:, F]

            if self.mesh.dim() == 2:
                mx = 0.5*(self.mesh.p[0, self.mesh.facets[0, F]] +
                          self.mesh.p[0, self.mesh.facets[1, F]])
                my = 0.5*(self.mesh.p[1, self.mesh.facets[0, F]] +
                          self.mesh.p[1, self.mesh.facets[1, F]])
                Fdofx = np.tile(mx, (Fdofs.shape[0], 1)).flatten()
                Fdofy = np.tile(my, (Fdofs.shape[0], 1)).flatten()
                locs = np.hstack((locs, np.vstack((Fdofx, Fdofy))))
            else:
                mx = np.sum(self.mesh.p[0, self.mesh.facets[:, F]], axis=0)/self.mesh.facets.shape[0]
                my = np.sum(self.mesh.p[1, self.mesh.facets[:, F]], axis=0)/self.mesh.facets.shape[0]
                mz = np.sum(self.mesh.p[2, self.mesh.facets[:, F]], axis=0)/self.mesh.facets.shape[0]
                Fdofx = np.tile(mx, (Fdofs.shape[0], 1)).flatten()
                Fdofy = np.tile(my, (Fdofs.shape[0], 1)).flatten()
                Fdofz = np.tile(mz, (Fdofs.shape[0], 1)).flatten()
                locs = np.hstack((locs, np.vstack((Fdofx, Fdofy, Fdofz))))

            dofs = np.hstack((dofs, Fdofs.flatten()))

        if check_edges:
            # handle edges
            if self.mesh.dim() == 3:
                E = self.mesh.edges_satisfying(test)
                if boundary:
                    E = np.intersect1d(E, self.mesh.boundary_edges())
                if dofrows is None:
                    Edofs = self.dofnum.e_dof[:, E]
                else:
                    Edofs = self.dofnum.e_dof[dofrows][:, E]

                mx = 0.5*(self.mesh.p[0, self.mesh.edges[0, E]] +
                          self.mesh.p[0, self.mesh.edges[1, E]])
                my = 0.5*(self.mesh.p[1, self.mesh.edges[0, E]] +
                          self.mesh.p[1, self.mesh.edges[1, E]])
                mz = 0.5*(self.mesh.p[2, self.mesh.edges[0, E]] +
                          self.mesh.p[2, self.mesh.edges[1, E]])

                Edofx = np.tile(mx, (Edofs.shape[0], 1)).flatten()
                Edofy = np.tile(my, (Edofs.shape[0], 1)).flatten()
                Edofz = np.tile(mz, (Edofs.shape[0], 1)).flatten()

                locs = np.hstack((locs, np.vstack((Edofx, Edofy, Edofz))))

                dofs = np.hstack((dofs, Edofs.flatten()))

        if self.mesh.dim() == 2:
            x[dofs] = bc(locs[0, :], locs[1, :])
        elif self.mesh.dim() == 3:
            x[dofs] = bc(locs[0, :], locs[1, :], locs[2, :])
        else:
            raise NotImplementedError("Method find_dofs not implemented " +
                                      "for the given dimension.")

        return x, dofs


class FacetBasis(GlobalBasis):
    def __init__(self, mesh, elem, mapping, intorder, side=None, dofnum=None):
        super(FacetBasis, self).__init__(mesh, elem, mapping, intorder)
        if dofnum is not None:
            self.dofnum = dofnum
            self.Nbfun = self.dofnum.t_dof.shape[0]

        self.X, self.W = get_quadrature(self.brefdom, self.intorder)

        # triangles where the basis is evaluated
        if side is None:
            self.find = np.nonzero(self.mesh.f2t[1, :] == -1)[0]
            self.tind = self.mesh.f2t[0, self.find]
        elif side == 0 or side == 1:
            self.find = np.nonzero(self.mesh.f2t[1, :] != -1)[0]
            self.tind = self.mesh.f2t[side, self.find]
        else:
            raise Exception("Parameter side must be 0 or 1. Facet shares only two elements.")

        # boundary refdom to global facet
        x = self.mapping.G(self.X, find=self.find)
        # global facet to refdom facet
        Y = self.mapping.invF(x, tind=self.tind)

        if hasattr(mesh, 'normals'):
            self.normals = np.repeat(mesh.normals[:, :, None], len(self.W), axis=2)
        else:
            # construct normal vectors from side=0 always
            Y0 = self.mapping.invF(x, tind=self.mesh.f2t[0, self.find]) # TODO check why without this works also (Y0 = Y)
            self.normals = self.mapping.normals(Y0, self.mesh.f2t[0, self.find], self.find, self.mesh.t2f)

        self.nf = len(self.find)

        self.phi = self.init_gbasis(self.nf, len(self.W), self.elem.order[0])
        self.dphi = self.init_gbasis(self.nf, len(self.W), self.elem.order[1])

        for j in range(self.Nbfun):
            self.phi[j], self.dphi[j] = self.elem.gbasis(self.mapping, Y, j, self.tind)

        self.nelems = self.nf
        self.dx = np.abs(self.mapping.detDG(self.X, find=self.find)) * np.tile(self.W, (self.nelems, 1))

        self.dofnum.t_dof = self.dofnum.t_dof[:, self.tind] # TODO this is required for asm(). Check for other options.

    def default_parameters(self):
        return np.array([
            self.global_coordinates(),
            self.mesh_parameters(),
            self.normals,
        ])

    def global_coordinates(self):
        return self.mapping.G(self.X, find=self.find)

    def mesh_parameters(self):
        return np.abs(self.mapping.detDG(self.X, self.find)) ** (1.0 / (self.mesh.dim() - 1))


class InteriorBasis(GlobalBasis):
    def __init__(self, mesh, elem, mapping, intorder):
        super(InteriorBasis, self).__init__(mesh, elem, mapping, intorder)

        self.X, self.W = get_quadrature(self.refdom, self.intorder)

        self.phi = self.init_gbasis(self.nt, len(self.W), self.elem.order[0])
        self.dphi = self.init_gbasis(self.nt, len(self.W), self.elem.order[1])

        for j in range(self.Nbfun):
            self.phi[j], self.dphi[j] = self.elem.gbasis(self.mapping, self.X, j)

        self.nelems = self.nt
        self.dx = np.abs(self.mapping.detDF(self.X)) * np.tile(self.W, (self.nelems, 1))

    def default_parameters(self):
        return np.array([
            self.global_coordinates(),
            self.mesh_parameters(),
        ])

    def global_coordinates(self):
        return self.mapping.F(self.X)

    def mesh_parameters(self):
        return np.abs(self.mapping.detDF(self.X)) ** (1.0 / self.mesh.dim())

    def refinterp(self, interp, Nrefs=1):
        """Refine and interpolate (for plotting)."""
        # mesh reference domain, refine and take the vertices
        meshclass = type(self.mesh)
        m = meshclass(initmesh='refdom')
        m.refine(Nrefs)
        X = m.p

        # map vertices to global elements
        x = self.mapping.F(X)

        # interpolate some previous discrete function at the vertices
        # of the refined mesh
        w = 0.0*x[0]

        for j in range(self.Nbfun):
            phi, _ = self.elem.gbasis(self.mapping, X, j)
            w += interp[self.dofnum.t_dof[j, :]][:, None]*phi

        nt = self.nt
        t = np.tile(m.t, (1, nt))
        dt = np.max(t)
        t += (dt+1)*np.tile(np.arange(nt), (m.t.shape[0]*m.t.shape[1], 1)).flatten('F').reshape((-1, m.t.shape[0])).T

        if X.shape[0]==1:
            p = np.array([x.flatten()])
        else:
            p = x[0].flatten()
            for itr in range(len(x)-1):
                p = np.vstack((p, x[itr+1].flatten()))

        M = meshclass(p, t, validate=False)

        return M, w.flatten()


def asm(kernel, ubasis, vbasis=None, w=None, nthreads=1, assemble=True):
    """
    Assembly using a kernel function.

    Parameters
    ----------
    kernel : function handle
        See Examples.
    ubasis : InteriorBasis
    vbasis : (OPTIONAL) InteriorBasis
    w : (OPTIONAL) ndarray
        An array of ndarrays of size Nelems x Nqp.
    nthreads : (OPTIONAL, default=1) int
        Number of threads to use in assembly. This is only
        useful if kernel is numba function compiled with
        nogil = True.

    Examples
    --------

    Creating multithreadable kernel function.

    from numba import njit

    @njit(nogil=True)
    def assemble(A, ix, u, du, v, dv, w, dx):
        for k in range(ix.shape[0]):
            i, j = ix[k]
            A[i, j] = np.sum((du[j][0] * dv[i][0] +\
                              du[j][1] * dv[i][1] +\
                              du[j][2] * dv[i][2]) * dx, axis=1)
    assemble.bilinear = True
    """
    import threading
    from itertools import product

    if vbasis is None:
        vbasis = ubasis

    nt = ubasis.nelems
    dx = ubasis.dx

    if w is None:
        w = ubasis.default_parameters()

    if kernel.bilinear:
        # initialize COO data structures
        data = np.zeros((vbasis.Nbfun, ubasis.Nbfun, nt))
        rows = np.zeros(ubasis.Nbfun * vbasis.Nbfun * nt)
        cols = np.zeros(ubasis.Nbfun * vbasis.Nbfun * nt)

        # create sparse matrix indexing
        for j in range(ubasis.phi.shape[0]):
            for i in range(vbasis.phi.shape[0]):
                # find correct location in data,rows,cols
                ixs = slice(nt * (vbasis.Nbfun * j + i), nt * (vbasis.Nbfun * j + i + 1))
                rows[ixs] = vbasis.dofnum.t_dof[i, :]
                cols[ixs] = ubasis.dofnum.t_dof[j, :]

        # create indices for linear loop over local stiffness matrix
        ixs = [i for j, i in product(range(vbasis.phi.shape[0]), range(ubasis.phi.shape[0]))]
        jxs = [j for j, i in product(range(vbasis.phi.shape[0]), range(ubasis.phi.shape[0]))]
        indices = np.array([ixs, jxs]).T

        # split local stiffness matrix elements to threads
        threads = [threading.Thread(target=kernel, args=(data, ij,
                                                         ubasis.phi,
                                                         ubasis.dphi,
                                                         vbasis.phi,
                                                         vbasis.dphi, w, dx)) for ij
                   in np.array_split(indices, nthreads, axis=0)]

        # start threads and wait for finishing
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if assemble:
            K = coo_matrix((np.transpose(data, (1, 0, 2)).flatten('C'), (rows, cols)),
                              shape=(vbasis.dofnum.N, ubasis.dofnum.N))
            K.eliminate_zeros()
            return K.tocsr()
        else:
            return (np.transpose(data, (1, 0, 2)).flatten('C'), (rows, cols))
    else:
        data = np.zeros((vbasis.Nbfun, nt))
        rows = np.zeros(vbasis.Nbfun * nt)
        cols = np.zeros(vbasis.Nbfun * nt)

        for i in range(vbasis.Nbfun):
            # find correct location in data,rows,cols
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = vbasis.dofnum.t_dof[i, :]
            cols[ixs] = np.zeros(nt)

        indices = range(vbasis.phi.shape[0])

        threads = [threading.Thread(target=kernel, args=(data, ix, vbasis.phi, vbasis.dphi, w, dx)) for ix
                   in np.array_split(indices, nthreads, axis=0)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return coo_matrix((data.flatten('C'), (rows, cols)),
                          shape=(vbasis.dofnum.N, 1)).toarray().T[0]


class Dofnum(object):
    """Generate a global degree-of-freedom numbering for arbitrary mesh."""

    n_dof = np.array([]) #: Nodal DOFs
    e_dof = np.array([]) #: Edge DOFs (3D only)
    f_dof = np.array([]) #: Facet DOFs (corresponds to edges in 2D)
    i_dof = np.array([]) #: Interior DOFs
    t_dof = np.array([]) #: Global DOFs, number-of-dofs x number-of-triangles
    N = 0 #: Total number of DOFs

    def __init__(self, mesh, element):
        # vertex dofs
        self.n_dof = np.reshape(np.arange(element.nodal_dofs
                                          * mesh.p.shape[1],
                                          dtype=np.int64),
                                (element.nodal_dofs, mesh.p.shape[1]), order='F')
        offset = element.nodal_dofs*mesh.p.shape[1]

        # edge dofs
        if hasattr(mesh, 'edges'): # 3D mesh
            self.e_dof = np.reshape(np.arange(element.edge_dofs
                                              * mesh.edges.shape[1],
                                              dtype=np.int64),
                                    (element.edge_dofs, mesh.edges.shape[1]),
                                    order='F') + offset
            offset = offset + element.edge_dofs*mesh.edges.shape[1]

        # facet dofs
        if hasattr(mesh, 'facets'): # 2D or 3D mesh
            self.f_dof = np.reshape(np.arange(element.facet_dofs
                                              * mesh.facets.shape[1],
                                              dtype=np.int64),
                                    (element.facet_dofs, mesh.facets.shape[1]),
                                    order='F') + offset
            offset = offset + element.facet_dofs*mesh.facets.shape[1]

        # interior dofs
        self.i_dof = np.reshape(np.arange(element.interior_dofs
                                          * mesh.t.shape[1],
                                          dtype=np.int64),
                                (element.interior_dofs, mesh.t.shape[1]),
                                order='F') + offset

        # global numbering
        self.t_dof = np.zeros((0, mesh.t.shape[1]), dtype=np.int64)

        # nodal dofs
        for itr in range(mesh.t.shape[0]):
            self.t_dof = np.vstack((self.t_dof,
                                    self.n_dof[:, mesh.t[itr, :]]))

        # edge dofs (if 3D)
        if hasattr(mesh, 'edges'):
            for itr in range(mesh.t2e.shape[0]):
                self.t_dof = np.vstack((self.t_dof,
                                        self.e_dof[:, mesh.t2e[itr, :]]))

        # facet dofs (if 2D or 3D)
        if hasattr(mesh, 'facets'):
            for itr in range(mesh.t2f.shape[0]):
                self.t_dof = np.vstack((self.t_dof,
                                        self.f_dof[:, mesh.t2f[itr, :]]))

        self.t_dof = np.vstack((self.t_dof, self.i_dof))

        self.N = np.max(self.t_dof) + 1

    def complement_dofs(self, D):
        return np.setdiff1d(np.arange(self.N), D)

    def get_dofs(self, N=None, F=None, E=None, T=None):
        """Return global DOF numbers corresponding to each
        node(N), facet(F), edge(E) and triangle(T)."""
        dofs = np.zeros(0, dtype=np.int64)
        if N is not None:
            dofs = np.hstack((dofs, self.n_dof[:, N].flatten()))
        if F is not None:
            dofs = np.hstack((dofs, self.f_dof[:, F].flatten()))
        if E is not None:
            dofs = np.hstack((dofs, self.e_dof[:, E].flatten()))
        if T is not None:
            dofs = np.hstack((dofs, self.i_dof[:, T].flatten()))
        return dofs.flatten()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
