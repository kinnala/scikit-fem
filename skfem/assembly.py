# -*- coding: utf-8 -*-
"""
Assemblers transform bilinear forms into sparse matrices
and linear forms into vectors. Moreover, they are used
for computing local functionals over elements and for
assigning DOF's to different topological entities of
the mesh.

There are currently two major assembler types.
:class:`skfem.assembly.AssemblerLocal` uses finite elements
defined through reference element. :class:`skfem.assembly.AssemblerGlobal`
uses finite elements defined through DOF functionals. The latter
are more general but slower.

Examples
--------
Assemble the stiffness matrix related to
the Poisson problem using the piecewise linear elements.

.. code-block:: python

    from skfem.mesh import MeshTri
    from skfem.assembly import AssemblerLocal
    from skfem.element import ElementLocalTriP1

    m = MeshTri()
    for itr in range(3):
        m.refine()
    e = ElementLocalTriP1()
    a = AssemblerLocal(m, e)

    def bilinear_form(du, dv):
        return du[0]*dv[0] + du[1]*dv[1]

    K = a.iasm(bilinear_form)
"""
import numpy as np
import inspect
from scipy.sparse import coo_matrix

import skfem.mesh
import skfem.mapping
from skfem.quadrature import get_quadrature
from skfem.utils import const_cell, cell_shape

class Assembler(object):
    """Finite element assembler."""

    def __init__(self):
        raise NotImplementedError("Assembler metaclass cannot be initialized.")

    def essential_bc(self, test=None, bc=None, boundary=True, dofrows=None, check_vertices=True,
                  check_facets=True, check_edges=True):
        """Helper function for setting essential boundary conditions.

        Does not test for element interior DOFs since they are not typically included in
        boundary conditions! Uses dofnum of 'u' variable.

        Remark: More convenient replacement for Assembler.dofnum_u.getdofs().
        
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
        if self.mesh.dim() == 1:
            raise Exception("Assembler.find_dofs not implemented for 1D mesh.")

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

        x = np.zeros(self.dofnum_u.N)

        dofs = np.zeros(0, dtype=np.int64)
        locs = np.zeros((self.mesh.dim(), 0))
        
        if check_vertices:
            # handle nodes
            N = self.mesh.nodes_satisfying(test)
            if boundary:
                N = np.intersect1d(N, self.mesh.boundary_nodes())
            if dofrows is None:
                Ndofs = self.dofnum_u.n_dof[:, N]
            else:
                Ndofs = self.dofnum_u.n_dof[dofrows][:, N]

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
                Fdofs = self.dofnum_u.f_dof[:, F]
            else:
                Fdofs = self.dofnum_u.f_dof[dofrows][:, F]

            if self.mesh.dim() == 2:
                mx = 0.5*(self.mesh.p[0, self.mesh.facets[0, F]] +
                          self.mesh.p[0, self.mesh.facets[1, F]])
                my = 0.5*(self.mesh.p[1, self.mesh.facets[0, F]] +
                          self.mesh.p[1, self.mesh.facets[1, F]])
                Fdofx = np.tile(mx, (Fdofs.shape[0], 1)).flatten()
                Fdofy = np.tile(my, (Fdofs.shape[0], 1)).flatten()
                locs = np.hstack((locs, np.vstack((Fdofx, Fdofy))))
            else:
                mx = 0.3333333*(self.mesh.p[0, self.mesh.facets[0, F]] +
                                self.mesh.p[0, self.mesh.facets[1, F]] +
                                self.mesh.p[0, self.mesh.facets[2, F]])
                my = 0.3333333*(self.mesh.p[1, self.mesh.facets[0, F]] +
                                self.mesh.p[1, self.mesh.facets[1, F]] +
                                self.mesh.p[1, self.mesh.facets[2, F]])
                mz = 0.3333333*(self.mesh.p[2, self.mesh.facets[0, F]] +
                                self.mesh.p[2, self.mesh.facets[1, F]] +
                                self.mesh.p[2, self.mesh.facets[2, F]])
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
                    Edofs = self.dofnum_u.e_dof[:, E]
                else:
                    Edofs = self.dofnum_u.e_dof[dofrows][:, E]

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
            raise NotImplementedError("Method essential_bc's not implemented for the given dimension.")

        return x, dofs

    def refinterp(self, interp, Nrefs=1):
        """Refine and interpolate (for plotting)."""
        # mesh reference domain, refine and take the vertices
        meshclass = type(self.mesh)
        m = meshclass(initmesh='refdom')
        m.refine(Nrefs)
        X = m.p

        # map vertices to global elements
        x = self.mapping.F(X)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]

        # interpolate some previous discrete function at the vertices
        # of the refined mesh
        w = 0.0*x[0]
        for j in range(Nbfun_u):
            phi, _ = self.elem_u.lbasis(X, j)
            w += np.outer(interp[self.dofnum_u.t_dof[j, :]], phi)

        nt = self.mesh.t.shape[1]
        t = np.tile(m.t, (1, nt))
        dt = np.max(t)
        t += (dt+1)*np.tile(np.arange(nt), (m.t.shape[0]*m.t.shape[1], 1)).flatten('F').reshape((-1, m.t.shape[0])).T

        p = x[0].flatten()
        for itr in range(len(x)-1):
            p = np.vstack((p, x[itr+1].flatten()))

        M = meshclass(p, t, validate=False)

        return M, w.flatten()


    def fillargs(self, oldform, newargs):
        """Used for filling functions with required set of arguments."""
        oldargs = inspect.getargspec(oldform).args
        if oldargs == newargs:
            # the given form already has correct arguments
            return oldform

        y = []
        for oarg in oldargs:
            # add corresponding new argument index to y for
            # each old argument
            for ix, narg in enumerate(newargs):
                if oarg == narg:
                    y.append(ix)
                    break

        if len(oldargs) == 1:
            def newform(*x):
                return oldform(x[y[0]])
        elif len(oldargs) == 2:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]])
        elif len(oldargs) == 3:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]])
        elif len(oldargs) == 4:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]])
        elif len(oldargs) == 5:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]])
        elif len(oldargs) == 6:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]])
        elif len(oldargs) == 7:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]])
        elif len(oldargs) == 8:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]])
        elif len(oldargs) == 9:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]])
        elif len(oldargs) == 10:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]])
        elif len(oldargs) == 11:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]],
                               x[y[10]])
        elif len(oldargs) == 12:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]],
                               x[y[10]], x[y[11]])
        elif len(oldargs) == 13:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]],
                               x[y[10]], x[y[11]], x[y[12]])
        elif len(oldargs) == 14:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]],
                               x[y[10]], x[y[11]], x[y[12]], x[y[13]])
        elif len(oldargs) == 15:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]],
                               x[y[10]], x[y[11]], x[y[12]], x[y[13]],
                               x[y[14]])
        elif len(oldargs) == 16:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]],
                               x[y[10]], x[y[11]], x[y[12]], x[y[13]],
                               x[y[14]], x[y[15]])
        else:
            raise NotImplementedError("Maximum number of arguments reached.")

        return newform


class AssemblerGlobal(Assembler):
    """An assembler for elements of type
    :class:`skfem.element.ElementGlobal`.

    These elements are defined through global degrees of freedom but
    are not limited to H^1-conforming elements. As a result,
    this assembler is more computationally intensive than
    :class:`skfem.assembly.AssemblerLocal` so use it instead
    if possible.

    Parameters
    ----------
    mesh : :class:`skfem.mesh.Mesh`
        The finite element mesh.

    elem_u : :class:`skfem.element.Element`
        The element for the solution function.

    elem_v : (OPTIONAL) :class:`skfem.element.Element`
        The element for the test function. By default, same element
        is used for both.

    intorder : (OPTIONAL) int
        The used quadrature order.

        The basis functions at quadrature points are precomputed
        in initializer. By default, the order of quadrature rule
        is deduced from the maximum polynomial degree of an element.
    """
    def __init__(self, mesh, elem_u, elem_v=None, intorder=None):
        if not isinstance(mesh, skfem.mesh.Mesh):
            raise Exception("First parameter must be an instance of "
                            "skfem.mesh.Mesh.")
        if not isinstance(elem_u, skfem.element.ElementGlobal):
            raise Exception("Second parameter must be an instance of "
                            "skfem.element.ElementGlobal.")
        if elem_v is not None:
            if not isinstance(elem_v, skfem.element.ElementGlobal):
                raise Exception("Third parameter must be an instance of "
                                "skfem.element.ElementGlobal.")

        self.mesh = mesh
        self.elem_u = elem_u
        self.dofnum_u = Dofnum(mesh, elem_u)
        self.mapping = mesh.mapping()

        # duplicate test function element type if None is given
        if elem_v is None:
            self.elem_v = elem_u
            self.dofnum_v = self.dofnum_u
        else:
            self.elem_v = elem_v
            self.dofnum_v = Dofnum(mesh, elem_v)

        if intorder is None:
            # compute the maximum polynomial degree from elements
            self.intorder = self.elem_u.maxdeg + self.elem_v.maxdeg
        else:
            self.intorder = intorder

        # quadrature points and weights
        X, _ = get_quadrature(self.mesh.refdom, self.intorder)
        # global quadrature points
        x = self.mapping.F(X, range(self.mesh.t.shape[1]))
        # pre-compute basis functions at quadrature points
        self.u, self.du, self.ddu, self.d4u = self.elem_u.evalbasis(self.mesh, x)

        if elem_v is None:
            self.v = self.u
            self.dv = self.du
            self.ddv = self.ddu
            self.d4v = self.d4u
        else:
            self.v, self.dv, self.ddv, self.d4v = self.elem_v.evalbasis(self.mesh, x)

    def iasm(self, form, tind=None, interp=None):
        if tind is None:
            # assemble on all elements by default
            tind = np.arange(self.mesh.t.shape[1],dtype=np.int64)
            indic = 1.0
        else:
            # indicator function for multiplying the end result
            indic = np.zeros(self.mesh.t.shape[1])
            indic[tind] = 1.0
            indic = indic[:, None]
            tind = np.arange(self.mesh.t.shape[1],dtype=np.int64)
        nt = len(tind)

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams or 'ddu' in oldparams or 'd4u' in oldparams:
            paramlist = ['u', 'v', 'du', 'dv', 'ddu', 'ddv', 'd4u', 'd4v', 'x', 'w', 'dw', 'ddw', 'd4w', 'h']
            bilinear = True
        else:
            paramlist = ['v', 'dv', 'ddv', 'd4v', 'x', 'w', 'dw', 'ddw', 'd4w', 'h']
            bilinear = False
        fform = self.fillargs(form, paramlist)

        # quadrature points and weights
        X, W = get_quadrature(self.mesh.refdom, self.intorder)

        # global quadrature points
        x = self.mapping.F(X, tind)

        # jacobian at quadrature points
        detDF = self.mapping.detDF(X, tind)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        Nbfun_v = self.dofnum_v.t_dof.shape[0]

        # interpolate some previous discrete function at quadrature points
        dim = self.mesh.p.shape[0]
        zero = 0.0*x[0]
        w, dw, ddw, d4w = ({} for i in range(4))
        if interp is not None:
            if not isinstance(interp, dict):
                raise Exception("The input solution vector(s) must be in a "
                                "dictionary! Pass e.g. {0:u} instead of u.")
            for k in interp:
                w[k] = zero
                dw[k] = const_cell(zero, dim)
                ddw[k] = const_cell(zero, dim, dim)
                d4w[k] = const_cell(zero, dim, dim)
                for j in range(Nbfun_u):
                    jdofs = self.dofnum_u.t_dof[j, :]
                    w[k] += interp[k][jdofs][:, None]\
                            * self.u[j]
                    for a in range(dim):
                        dw[k][a] += interp[k][jdofs][:, None]\
                                    * self.du[j][a]
                        for b in range(dim):
                            ddw[k][a][b] += interp[k][jdofs][:, None]\
                                            * self.ddu[j][a][b]
                            d4w[k][a][b] += interp[k][jdofs][:, None]\
                                            * self.d4u[j][a][b]

        # compute the mesh parameter from jacobian determinant
        h = np.abs(detDF)**(1.0/self.mesh.dim())

        # bilinear form
        if bilinear:
            # initialize sparse matrix structures
            data = np.zeros(Nbfun_u*Nbfun_v*nt)
            rows = np.zeros(Nbfun_u*Nbfun_v*nt)
            cols = np.zeros(Nbfun_u*Nbfun_v*nt)

            for j in range(Nbfun_u):
                for i in range(Nbfun_v):
                    # find correct location in data,rows,cols
                    ixs = slice(nt*(Nbfun_v*j + i), nt*(Nbfun_v*j + i + 1))

                    # compute entries of local stiffness matrices
                    data[ixs] = np.dot(fform(self.u[j], self.v[i],
                                            self.du[j], self.dv[i],
                                            self.ddu[j], self.ddv[i],
                                            self.d4u[j], self.d4v[i],
                                            x, w, dw, ddw, d4w, h)*indic*np.abs(detDF), W)
                    rows[ixs] = self.dofnum_v.t_dof[i, tind]
                    cols[ixs] = self.dofnum_u.t_dof[j, tind]

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, self.dofnum_u.N)).tocsr()

        else:
            # initialize sparse matrix structures
            data = np.zeros(Nbfun_v*nt)
            rows = np.zeros(Nbfun_v*nt)
            cols = np.zeros(Nbfun_v*nt)

            for i in range(Nbfun_v):
                # find correct location in data,rows,cols
                ixs = slice(nt*i, nt*(i + 1))

                # compute entries of local stiffness matrices
                data[ixs] = np.dot(fform(self.v[i], self.dv[i], self.ddv[i], self.d4v[i],
                                         x, w, dw, ddw, d4w, h)*indic*np.abs(detDF), W)
                rows[ixs] = self.dofnum_v.t_dof[i, :]
                cols[ixs] = np.zeros(nt)

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, 1)).toarray().T[0]

    def fasm(self, form, intorder=None, interior=False, normals=True):
        if interior:
            # assemble on all interior facets
            find = self.mesh.interior_facets()
        else:
            # assemble on all boundary facets
            find = self.mesh.boundary_facets()

        ne = find.shape[0]

        if intorder is None:
            intorder = 2*self.elem_u.maxdeg

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        if 'u1' in oldparams or 'du1' in oldparams or 'ddu1' in oldparams:
            bilinear = True
            if interior is False:
                raise Exception("The supplied form contains u1 although "
                                "no interior=True is given.")
        else:
            if 'u' in oldparams or 'du' in oldparams or 'ddu' in oldparams:
                bilinear = True
            else:
                bilinear = False
        if interior:
            if bilinear:
                paramlist = ['u1', 'u2', 'du1', 'du2', 'ddu1', 'ddu2',
                             'v1', 'v2', 'dv1', 'dv2', 'ddv1', 'ddv2',
                             'x', 'n', 't', 'h']
            else:
                paramlist = ['v1', 'v2', 'dv1', 'dv2', 'ddv1', 'ddv2',
                             'x', 'n', 't', 'h']
        else:
            if bilinear:
                paramlist = ['u', 'du', 'ddu', 'v', 'dv', 'ddv',
                             'x', 'n', 't', 'h']
            else:
                paramlist = ['v', 'dv', 'ddv',
                             'x', 'n', 't', 'h']

        fform = self.fillargs(form, paramlist)

        X, W = get_quadrature(self.mesh.brefdom, intorder)

        # indices of elements at different sides of facets
        tind1 = self.mesh.f2t[0, find]
        tind2 = self.mesh.f2t[1, find]

        # mappings
        x = self.mapping.G(X, find=find) # reference facet to global facet
        detDG = self.mapping.detDG(X, find)

        # compute basis function values at quadrature points
        u1, du1, ddu1 = self.elem_u.evalbasis(self.mesh, x, tind=tind1)
        if interior:
            u2, du2, ddu2 = self.elem_u.evalbasis(self.mesh, x, tind=tind2)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        Nbfun_v = self.dofnum_v.t_dof.shape[0]
        dim = self.mesh.p.shape[0]

        n = {}
        t = {}
        if normals:
            Y = self.mapping.invF(x, tind=tind1) # global facet to ref element
            n = self.mapping.normals(Y, tind1, find, self.mesh.t2f)
            if len(n) == 2:
                t[0] = -n[1]
                t[1] = n[0]

        # TODO function interpolation

        h = np.abs(detDG)**(1.0/(self.mesh.dim()-1.0))

        # bilinear form
        if bilinear:
            # TODO
            raise NotImplementedError("bilinear facet assembly not implemented")
        # linear form
        else:
            if interior:
                ndata = Nbfun_v*ne
                data = np.zeros(2*ndata)
                rows = np.zeros(2*ndata)
                cols = np.zeros(2*ndata)

                for i in range(Nbfun_v):
                    if i == 0:
                        # these are zeros corresponding to the shapes of u,du
                        z = const_cell(0, *cell_shape(u1))
                        dz = const_cell(0, *cell_shape(du1))
                        ddz = const_cell(0, *cell_shape(ddu1))
                    ixs1 = slice(ne*i,
                                 ne*(i + 1))
                    ixs2 = slice(ne*i + ndata,
                                 ne*(i + 1) + ndata)

                    data[ixs1] = np.dot(fform(u1[i], z[i], du1[i], dz[i], ddu1[i], ddz[i], x, n, t, h)*np.abs(detDG), W)
                    rows[ixs1] = self.dofnum_v.t_dof[i, tind1]
                    cols[ixs1] = np.zeros(ne)

                    data[ixs2] = np.dot(fform(z[i], u2[i], dz[i], du2[i], ddz[i], ddu2[i], x, n, t, h)*np.abs(detDG), W)
                    rows[ixs2] = self.dofnum_v.t_dof[i, tind2]
                    cols[ixs2] = np.zeros(ne)

                return coo_matrix((data, (rows, cols)),
                                  shape=(self.dofnum_v.N, 1)).toarray().T[0]
            else:
                # TODO
                raise NotImplementedError("linear + boundary not implemented")

    def inorm(self, form, interp, intorder=None):
        """Evaluate L2-norms of solution vectors inside elements. Useful for
        e.g. evaluating a posteriori estimators.

        Parameters
        ----------
        form : function handle
            The function for which the L2 norm is evaluated. Can consist
            of the following

            +-----------+----------------------+
            | Parameter | Explanation          |
            +-----------+----------------------+
            | u         | solution             |
            +-----------+----------------------+
            | du        | solution derivatives |
            +-----------+----------------------+
            | ddu       | -||- 2nd derivatives |
            +-----------+----------------------+
            | d4u       | -||- 4th derivatives |
            +-----------+----------------------+
            | x         | spatial location     |
            +-----------+----------------------+
            | h         | the mesh parameter   |
            +-----------+----------------------+

            The function handle must use these exact names for
            the variables. Unused variable names can be omitted.

        interp : dict of numpy arrays
            The solutions that are interpolated.

        intorder : (OPTIONAL) int
            The order of polynomials for which the applied
            quadrature rule is exact. By default,
            2*Element.maxdeg is used.
        """
        # evaluate norm on all elements
        tind = range(self.mesh.t.shape[1])

        if not isinstance(interp, dict):
            raise Exception("The input solution vector(s) must be in a "
                            "dictionary! Pass e.g. {0:u} instead of u.")

        if intorder is None:
            intorder = 2*self.elem_u.maxdeg
        intorder=self.intorder

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        paramlist = ['u', 'du', 'ddu', 'd4u', 'x', 'h']
        fform = self.fillargs(form, paramlist)

        X, W = get_quadrature(self.mesh.refdom, intorder)

        # mappings
        x = self.mapping.F(X, tind) # reference facet to global facet

        # jacobian at quadrature points
        detDF = self.mapping.detDF(X, tind)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        dim = self.mesh.p.shape[0]

        # interpolate the solution vectors at quadrature points
        zero = 0.0*x[0]
        w, dw, ddw, d4w = ({} for i in range(4))
        for k in interp:
            w[k] = zero
            dw[k] = const_cell(zero, dim)
            ddw[k] = const_cell(zero, dim, dim)
            d4w[k] = const_cell(zero, dim, dim)
            for j in range(Nbfun_u):
                jdofs = self.dofnum_u.t_dof[j, :]
                w[k] += interp[k][jdofs][:, None]\
                        * self.u[j]
                for a in range(dim):
                    dw[k][a] += interp[k][jdofs][:, None]\
                                * self.du[j][a]
                    for b in range(dim):
                        ddw[k][a][b] += interp[k][jdofs][:, None]\
                                        * self.ddu[j][a][b]
                        d4w[k][a][b] += interp[k][jdofs][:, None]\
                                        * self.d4u[j][a][b]

        # compute the mesh parameter from jacobian determinant
        h = np.abs(detDF)**(1.0/self.mesh.dim())

        return np.dot(fform(w, dw, ddw, d4w, x, h)**2*np.abs(detDF), W)

    def fnorm(self, form, interp, intorder=None, interior=False, normals=True):
        if interior:
            # evaluate norm on all interior facets
            find = self.mesh.interior_facets()
        else:
            # evaluate norm on all boundary facets
            find = self.mesh.boundary_facets()

        if not isinstance(interp, dict):
            raise Exception("The input solution vector(s) must be in a "
                            "dictionary! Pass e.g. {0:u} instead of u.")

        if intorder is None:
            intorder = 2*self.elem_u.maxdeg

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        if 'u1' in oldparams or 'du1' in oldparams or 'ddu1' in oldparams:
            if interior is False:
                raise Exception("The supplied form contains u1 although "
                                "no interior=True is given.")
        if interior:
            paramlist = ['u1', 'u2', 'du1', 'du2', 'ddu1', 'ddu2',
                         'x', 'n', 't', 'h']
        else:
            paramlist = ['u', 'du', 'ddu', 'x', 'n', 't', 'h']
        fform = self.fillargs(form, paramlist)

        X, W = get_quadrature(self.mesh.brefdom, intorder)

        # indices of elements at different sides of facets
        tind1 = self.mesh.f2t[0, find]
        tind2 = self.mesh.f2t[1, find]

        # mappings
        x = self.mapping.G(X, find=find) # reference facet to global facet
        detDG = self.mapping.detDG(X, find)

        # compute basis function values at quadrature points
        u1, du1, ddu1, _ = self.elem_u.evalbasis(self.mesh, x, tind=tind1)
        if interior:
            u2, du2, ddu2, _ = self.elem_u.evalbasis(self.mesh, x, tind=tind2)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        dim = self.mesh.p.shape[0]

        n = {}
        t = {}
        if normals:
            Y = self.mapping.invF(x, tind=tind1) # global facet to ref element
            n = self.mapping.normals(Y, tind1, find, self.mesh.t2f)
            if len(n) == 2:
                t[0] = -n[1]
                t[1] = n[0]

        # interpolate the solution vectors at quadrature points
        zero = np.zeros((len(find), len(W)))
        w1, dw1, ddw1 = ({} for i in range(3))
        if interior:
            w2, dw2, ddw2 = ({} for i in range(3))
        for k in interp:
            w1[k] = zero
            dw1[k] = const_cell(zero, dim)
            ddw1[k] = const_cell(zero, dim, dim)
            if interior:
                w2[k] = zero
                dw2[k] = const_cell(zero, dim)
                ddw2[k] = const_cell(zero, dim, dim)
            for j in range(Nbfun_u):
                jdofs1 = self.dofnum_u.t_dof[j, tind1]
                jdofs2 = self.dofnum_u.t_dof[j, tind2]
                w1[k] += interp[k][jdofs1][:, None] * u1[j]
                if interior:
                    w2[k] += interp[k][jdofs2][:, None] * u2[j]
                for a in range(dim):
                    dw1[k][a] += interp[k][jdofs1][:, None]\
                                 * du1[j][a]
                    if interior:
                        dw2[k][a] += interp[k][jdofs2][:, None]\
                                     * du2[j][a]
                    for b in range(dim):
                        ddw1[k][a][b] += interp[k][jdofs1][:, None]\
                                         * ddu1[j][a][b]
                        if interior:
                            ddw2[k][a][b] += interp[k][jdofs2][:, None]\
                                             * ddu2[j][a][b]

        h = np.abs(detDG)**(1.0/(self.mesh.dim()-1.0))

        if interior:
            return np.dot(fform(w1, w2, dw1, dw2, ddw1, ddw2,
                                x, n, t, h)**2*np.abs(detDG), W), find
        else:
            return np.dot(fform(w1, dw1, ddw1,
                                x, n, t, h)**2*np.abs(detDG), W), find

class AssemblerLocalMortar(Assembler):
    """
    For assembling couplings on interfaces
    """
    def __init__(self, mortar, mesh1, mesh2, elem1, elem2=None):
        if not isinstance(mesh1, skfem.mesh.Mesh):
            raise Exception("Wrong parameter type")
        if not isinstance(mesh2, skfem.mesh.Mesh):
            raise Exception("Wrong parameter type")
        if not isinstance(elem1, skfem.element.ElementLocal):
            raise Exception("Wrong parameter type")

        self.mortar = mortar
        self.mortar_mapping = mortar.mapping()

        # get default mapping from the mesh
        self.mapping1 = mesh1.mapping()
        self.mapping2 = mesh2.mapping()

        self.mesh1 = mesh1
        self.mesh2 = mesh2
        self.elem1 = elem1
        self.dofnum1 = Dofnum(mesh1, elem1)

        # duplicate test function element type if None is given
        if elem2 is None:
            self.elem2 = elem1
            self.dofnum2 = Dofnum(mesh2, elem1)
        else:
            self.elem2 = elem2
            self.dofnum2 = Dofnum(mesh2, elem2)

    def fasm(self, form, find=None, intorder=None):
        """Facet assembly."""

        find1 = self.mortar.f2t[0, :]
        find2 = self.mortar.f2t[1, :]

        if intorder is None:
            intorder = self.elem1.maxdeg + self.elem2.maxdeg

        ne = find1.shape[0]

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args

        if 'u1' in oldparams or 'du1' in oldparams:
            paramlist = ['u1', 'u2', 'v1', 'v2',
                         'du1', 'du2', 'dv1', 'dv2',
                         'x', 'h', 'n']
            bilinear = True
        else:
            if 'v1' not in oldparams or 'dv1' not in oldparams:
                raise Exception("Invalid form")
            paramlist = ['v1', 'v2', 'dv1', 'dv2',
                         'x', 'h', 'n']
            bilinear = False

        fform = self.fillargs(form, paramlist)

        X, W = get_quadrature(self.mesh1.brefdom, intorder)

        # boundary element indices
        tind1 = self.mesh1.f2t[0, find1]
        tind2 = self.mesh2.f2t[0, find2]

        # mappings
        x = self.mortar_mapping.G(X) # reference facet to global facet
      
        Y1 = self.mapping1.invF(x, tind=tind1) # global facet to ref element
        Y2 = self.mapping2.invF(x, tind=tind2) # global facet to ref element

        Nbfun1 = self.dofnum1.t_dof.shape[0]
        Nbfun2 = self.dofnum2.t_dof.shape[0]

        detDG = self.mortar_mapping.detDG(X)

        # compute normal vectors
        n = {}
        # normals based on tind1 only
        n = self.mapping1.normals(Y1, tind1, find1, self.mesh1.t2f)

        # compute the mesh parameter from jacobian determinant
        h = np.abs(detDG)**(1.0/(self.mesh1.dim() - 1.0))

        # initialize sparse matrix structures
        ndata = Nbfun1*Nbfun2*ne
        data = np.zeros(4*ndata)
        rows = np.zeros(4*ndata)
        cols = np.zeros(4*ndata)

        for j in range(Nbfun1):
            u1, du1 = self.elem1.gbasis(self.mapping1, Y1, j, tind1)
            u2, du2 = self.elem2.gbasis(self.mapping2, Y2, j, tind2)
            if j == 0:
                # these are zeros corresponding to the shapes of u,du
                z = const_cell(0, *cell_shape(u2))
                dz = const_cell(0, *cell_shape(du2))
            for i in range(Nbfun2):
                v1, dv1 = self.elem2.gbasis(self.mapping1, Y1, i, tind1)
                v2, dv2 = self.elem2.gbasis(self.mapping2, Y2, i, tind2)

                ixs1 = slice(ne*(Nbfun2*j + i),
                             ne*(Nbfun2*j + i + 1))
                ixs2 = slice(ne*(Nbfun2*j + i) + ndata,
                             ne*(Nbfun2*j + i + 1) + ndata)
                ixs3 = slice(ne*(Nbfun2*j + i) + 2*ndata,
                             ne*(Nbfun2*j + i + 1) + 2*ndata)
                ixs4 = slice(ne*(Nbfun2*j + i) + 3*ndata,
                             ne*(Nbfun2*j + i + 1) + 3*ndata)

                data[ixs1] = np.dot(fform(u1, z, v1, z,
                                          du1, dz, dv1, dz,
                                          x, h, n)*np.abs(detDG), W)
                rows[ixs1] = self.dofnum1.t_dof[i, tind1]
                cols[ixs1] = self.dofnum1.t_dof[j, tind1]

                data[ixs2] = np.dot(fform(z, u2, z, v2,
                                          dz, du2, dz, dv2,
                                          x, h, n)*np.abs(detDG), W)
                rows[ixs2] = self.dofnum2.t_dof[i, tind2] + self.dofnum1.N
                cols[ixs2] = self.dofnum2.t_dof[j, tind2] + self.dofnum1.N

                data[ixs3] = np.dot(fform(z, u2, v1, z,
                                          dz, du2, dv1, dz,
                                          x, h, n)*np.abs(detDG), W)
                rows[ixs3] = self.dofnum1.t_dof[i, tind1]
                cols[ixs3] = self.dofnum2.t_dof[j, tind2] + self.dofnum1.N

                data[ixs4] = np.dot(fform(u1, z, z, v2,
                                          du1, dz, dz, dv2,
                                          x, h, n)*np.abs(detDG), W)
                rows[ixs4] = self.dofnum2.t_dof[i, tind2] + self.dofnum1.N
                cols[ixs4] = self.dofnum1.t_dof[j, tind1]
                #ixs = slice(ne*(Nbfun_v*j + i), ne*(Nbfun_v*j + i + 1))

                #data[ixs] = np.dot(fform(u, v, du, dv,
                #                         x, h, n)*np.abs(detDG), W)
                #rows[ixs] = self.dofnum_v.t_dof[i, tind2]
                #cols[ixs] = self.dofnum_u.t_dof[j, tind1]

        return coo_matrix((data, (rows, cols)),
                          shape=(self.dofnum1.N + self.dofnum2.N, self.dofnum1.N + self.dofnum2.N)).tocsr()


class AssemblerLocal(Assembler):
    """An assembler for Element classes.

    These elements are defined through reference elements
    and are limited to H^1-conforming elements.

    Parameters
    ----------
    mesh : :class:`skfem.mesh.Mesh`
        The finite element mesh.

    elem_u : :class:`skfem.element.Element`
        The element for the solution function.

    elem_v : (OPTIONAL) :class:`skfem.element.Element`
        The element for the test function. By default,
        the same element is used for both.

    mapping : (OPTIONAL) :class:`skfem.mapping.Mapping`
        The mesh will give some sort of default mapping but sometimes, e.g.
        when using isoparametric elements, the user might have to provide
        a different mapping.
    """
    def __init__(self, mesh, elem_u, elem_v=None, mapping=None):
        if not isinstance(mesh, skfem.mesh.Mesh):
            raise Exception("First parameter must be an instance of "
                            "skfem.mesh.Mesh!")
        if not isinstance(elem_u, skfem.element.ElementLocal):
            raise Exception("Second parameter must be an instance of "
                            "skfem.element.Element!")

        # get default mapping from the mesh
        if mapping is None:
            self.mapping = mesh.mapping()
        else:
            self.mapping = mapping # assumes an already initialized mapping

        self.mesh = mesh
        self.elem_u = elem_u
        self.dofnum_u = Dofnum(mesh, elem_u)

        # duplicate test function element type if None is given
        if elem_v is None:
            self.elem_v = elem_u
            self.dofnum_v = self.dofnum_u
        else:
            self.elem_v = elem_v
            self.dofnum_v = Dofnum(mesh, elem_v)

    def iasm(self, form, intorder=None, tind=None, interp=None, precompute_basis=False):
        """Return a matrix related to a bilinear or linear form
        where the integral is over the interior of the domain.

        Parameters
        ----------
        form : function handle
            The bilinear or linear form function handle.
            The supported parameters can be found in the
            following table.

            +-----------+----------------------+--------------+
            | Parameter | Explanation          | Supported in |
            +-----------+----------------------+--------------+
            | u         | solution             | bilinear     |
            +-----------+----------------------+--------------+
            | v         | test fun             | both         |
            +-----------+----------------------+--------------+
            | du        | solution derivatives | bilinear     |
            +-----------+----------------------+--------------+
            | dv        | test fun derivatives | both         |
            +-----------+----------------------+--------------+
            | x         | spatial location     | both         |
            +-----------+----------------------+--------------+
            | w         | cf. interp           | both         |
            +-----------+----------------------+--------------+
            | h         | the mesh parameter   | both         |
            +-----------+----------------------+--------------+

            The function handle must use these exact names for
            the variables. Unused variable names can be omitted.

            Examples of valid bilinear forms:
            ::

                def bilin_form1(du, dv):
                    # Note that the element must be
                    # defined for two-dimensional
                    # meshes for this to make sense!
                    return du[0]*dv[0] + du[1]*dv[1]

                def bilin_form2(du, v):
                    return du[0]*v

                bilin_form3 = lambda u, v, x: x[0]**2*u*v

            Examples of valid linear forms:
            ::

                def lin_form1(v):
                    return v

                def lin_form2(h, x, v):
                    import numpy as np
                    mesh_parameter = h
                    X = x[0]
                    Y = x[1]
                    return mesh_parameter*np.sin(np.pi*X)*np.sin(np.pi*Y)*v

            The linear forms are automatically detected to be
            non-bilinear through the omission of u or du.

        intorder : (OPTIONAL) int
            The order of polynomials for which the applied
            quadrature rule is exact. By default,
            2*Element.maxdeg is used. Reducing this number
            can sometimes reduce the computation time.

        interp : (OPTIONAL) numpy array
            Using this flag, the user can provide
            a solution vector that is interpolated
            to the quadrature points and included in
            the computation of the bilinear form
            (the variable w). Useful e.g. when solving
            nonlinear problems.

        tind : (OPTIONAL) numpy array
            The indices of elements that are integrated over.
            By default, all elements of the mesh are included.

        precompute_basis : (OPTIONAL, default: False) boolean
            This will usually make assembly faster but requires
            more memory (approximately no-basis-functions/2
            times more memory).
        """
        if tind is None:
            # assemble on all elements by default
            tind = range(self.mesh.t.shape[1])
        nt = len(tind)
        if intorder is None:
            # compute the maximum polynomial degree from elements
            intorder = self.elem_u.maxdeg + self.elem_v.maxdeg

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams:
            paramlist = ['u', 'v', 'du', 'dv', 'x', 'w', 'h']
            bilinear = True
        else:
            paramlist = ['v', 'dv', 'x', 'w', 'h']
            bilinear = False
        fform = self.fillargs(form, paramlist)

        # quadrature points and weights
        X, W = get_quadrature(self.mesh.refdom, intorder)

        # global quadrature points
        x = self.mapping.F(X, tind)

        # jacobian at quadrature points
        detDF = self.mapping.detDF(X, tind)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        Nbfun_v = self.dofnum_v.t_dof.shape[0]

        # interpolate some previous discrete function at quadrature points
        w = {}
        if interp is not None:
            if not isinstance(interp, dict):
                raise Exception("The input solution vector(s) must be in a "
                                "dictionary! Pass e.g. {0:u} instead of u.")
            for k in interp:
                w[k] = 0.0*x[0]
                for j in range(Nbfun_u):
                    phi, _ = self.elem_u.lbasis(X, j)
                    w[k] += np.outer(interp[k][self.dofnum_u.t_dof[j, :]], phi)

        # compute the mesh parameter from jacobian determinant
        h = np.abs(detDF)**(1.0/self.mesh.dim())
        absdetDF = np.abs(detDF)

        if precompute_basis:
            U, dU = {}, {}
            for j in range(Nbfun_u):
                U[j], dU[j] = self.elem_u.gbasis(self.mapping, X, j, tind)

        # bilinear form
        if bilinear:
            # initialize sparse matrix structures
            data = np.zeros(Nbfun_u*Nbfun_v*nt)
            rows = np.zeros(Nbfun_u*Nbfun_v*nt)
            cols = np.zeros(Nbfun_u*Nbfun_v*nt)

            for j in range(Nbfun_u):
                if precompute_basis:
                    u, du = U[j], dU[j]
                else:
                    u, du = self.elem_u.gbasis(self.mapping, X, j, tind)
                for i in range(Nbfun_v):
                    if precompute_basis:
                        v, dv = U[i], dU[i]
                    else:
                        v, dv = self.elem_v.gbasis(self.mapping, X, i, tind)

                    # find correct location in data,rows,cols
                    ixs = slice(nt*(Nbfun_v*j+i), nt*(Nbfun_v*j+i+1))

                    # compute entries of local stiffness matrices
                    data[ixs] = np.dot(fform(u, v, du, dv, x, w, h)
                                       * absdetDF, W)
                    rows[ixs] = self.dofnum_v.t_dof[i, tind]
                    cols[ixs] = self.dofnum_u.t_dof[j, tind]

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, self.dofnum_u.N)).tocsr()

        else:
            # initialize sparse matrix structures
            data = np.zeros(Nbfun_v*nt)
            rows = np.zeros(Nbfun_v*nt)
            cols = np.zeros(Nbfun_v*nt)

            for i in range(Nbfun_v):
                if precompute_basis:
                    v, dv = U[i], dU[i]
                else:
                    v, dv = self.elem_v.gbasis(self.mapping, X, i, tind)

                # find correct location in data,rows,cols
                ixs = slice(nt*i, nt*(i+1))

                # compute entries of local stiffness matrices
                data[ixs] = np.dot(fform(v, dv, x, w, h)*absdetDF, W)
                rows[ixs] = self.dofnum_v.t_dof[i, tind]
                cols[ixs] = np.zeros(nt)

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, 1)).toarray().T[0]

    def fasm(self, form, find=None, interior=False, intorder=None,
             normals=True, interp=None):
        """Facet assembly."""
        if find is None:
            if interior:
                find = self.mesh.interior_facets()
            else:
                find = self.mesh.boundary_facets()

        if intorder is None:
            intorder = self.elem_u.maxdeg + self.elem_v.maxdeg

        nv = self.mesh.p.shape[1]
        nt = self.mesh.t.shape[1]
        ne = find.shape[0]

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        if interior:
            if 'u1' in oldparams or 'du1' in oldparams:
                paramlist = ['u1', 'u2', 'v1', 'v2',
                             'du1', 'du2', 'dv1', 'dv2',
                             'x', 'h', 'n', 'w', 'dw']
                bilinear = True
            else:
                paramlist = ['v1', 'v2', 'dv1', 'dv2',
                             'x', 'h', 'n', 'w', 'dw']
                bilinear = False
        else:
            if 'u' in oldparams or 'du' in oldparams:
                paramlist = ['u', 'v', 'du', 'dv', 'x', 'h', 'n', 'w', 'dw']
                bilinear = True
            else:
                paramlist = ['v', 'dv', 'x', 'h', 'n', 'w', 'dw']
                bilinear = False
        fform = self.fillargs(form, paramlist)

        X, W = get_quadrature(self.mesh.brefdom, intorder)

        # boundary element indices
        tind1 = self.mesh.f2t[0, find]
        tind2 = self.mesh.f2t[1, find]

        dim = self.mesh.p.shape[0]

        # mappings
        x = self.mapping.G(X, find=find) # reference facet to global facet
        Y1 = self.mapping.invF(x, tind=tind1) # global facet to ref element
        Y2 = self.mapping.invF(x, tind=tind2) # global facet to ref element

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        Nbfun_v = self.dofnum_v.t_dof.shape[0]

        detDG = self.mapping.detDG(X, find)

        # compute normal vectors
        n = {}
        if normals:
            # normals based on tind1 only
            n = self.mapping.normals(Y1, tind1, find, self.mesh.t2f)

        # compute the mesh parameter from jacobian determinant
        if self.mesh.dim() > 1.0:
            h = np.abs(detDG)**(1.0/(self.mesh.dim() - 1.0))
        else: # exception for 1D mesh (no boundary h defined)
            h = None

        # interpolate some previous discrete function at quadrature points
        w = {}
        dw = {}
        if interp is not None:
            if not isinstance(interp, dict):
                raise Exception("The input solution vector(s) must be in a "
                                "dictionary! Pass e.g. {0:u} instead of u.")
            for k in interp:
                w[k] = 0.0*x[0]
                dw[k] = const_cell(0.0*x[0], dim)
                for j in range(Nbfun_u):
                    phi, dphi = self.elem_u.gbasis(self.mapping, Y1, j, tind1)
                    w[k] += interp[k][self.dofnum_u.t_dof[j, tind1], None]*phi
                    for a in range(dim):
                        dw[k][a] += interp[k][self.dofnum_u.t_dof[j, tind1], None]*dphi[a]

        # bilinear form
        if bilinear:
            # initialize sparse matrix structures
            ndata = Nbfun_u*Nbfun_v*ne
            if interior:
                data = np.zeros(4*ndata)
                rows = np.zeros(4*ndata)
                cols = np.zeros(4*ndata)
            else:
                data = np.zeros(ndata)
                rows = np.zeros(ndata)
                cols = np.zeros(ndata)

            for j in range(Nbfun_u):
                u1, du1 = self.elem_u.gbasis(self.mapping, Y1, j, tind1)
                if interior:
                    u2, du2 = self.elem_u.gbasis(self.mapping, Y2, j, tind2)
                    if j == 0:
                        # these are zeros corresponding to the shapes of u,du
                        z = const_cell(0, *cell_shape(u2))
                        dz = const_cell(0, *cell_shape(du2))
                for i in range(Nbfun_v):
                    v1, dv1 = self.elem_v.gbasis(self.mapping, Y1, i, tind1)
                    if interior:
                        v2, dv2 = self.elem_v.gbasis(self.mapping, Y2, i, tind2)

                    # compute entries of local stiffness matrices
                    if interior:
                        ixs1 = slice(ne*(Nbfun_v*j + i),
                                     ne*(Nbfun_v*j + i + 1))
                        ixs2 = slice(ne*(Nbfun_v*j + i) + ndata,
                                     ne*(Nbfun_v*j + i + 1) + ndata)
                        ixs3 = slice(ne*(Nbfun_v*j + i) + 2*ndata,
                                     ne*(Nbfun_v*j + i + 1) + 2*ndata)
                        ixs4 = slice(ne*(Nbfun_v*j + i) + 3*ndata,
                                     ne*(Nbfun_v*j + i + 1) + 3*ndata)

                        data[ixs1] = np.dot(fform(u1, z, v1, z,
                                                  du1, dz, dv1, dz,
                                                  x, h, n, w, dw)*np.abs(detDG), W)
                        rows[ixs1] = self.dofnum_v.t_dof[i, tind1]
                        cols[ixs1] = self.dofnum_u.t_dof[j, tind1]

                        data[ixs2] = np.dot(fform(z, u2, z, v2,
                                                  dz, du2, dz, dv2,
                                                  x, h, n, w, dw)*np.abs(detDG), W)
                        rows[ixs2] = self.dofnum_v.t_dof[i, tind2]
                        cols[ixs2] = self.dofnum_u.t_dof[j, tind2]

                        data[ixs3] = np.dot(fform(z, u2, v1, z,
                                                  dz, du2, dv1, dz,
                                                  x, h, n, w, dw)*np.abs(detDG), W)
                        rows[ixs3] = self.dofnum_v.t_dof[i, tind1]
                        cols[ixs3] = self.dofnum_u.t_dof[j, tind2]

                        data[ixs4] = np.dot(fform(u1, z, z, v2,
                                                  du1, dz, dz, dv2,
                                                  x, h, n, w, dw)*np.abs(detDG), W)
                        rows[ixs4] = self.dofnum_v.t_dof[i, tind2]
                        cols[ixs4] = self.dofnum_u.t_dof[j, tind1]
                    else:
                        ixs = slice(ne*(Nbfun_v*j + i), ne*(Nbfun_v*j + i + 1))
                        data[ixs] = np.dot(fform(u1, v1, du1, dv1,
                                                 x, h, n, w, dw)*np.abs(detDG), W)
                        rows[ixs] = self.dofnum_v.t_dof[i, tind1]
                        cols[ixs] = self.dofnum_u.t_dof[j, tind1]

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, self.dofnum_u.N)).tocsr()

        # linear form
        else:
            if interior:
                # could not find any use case
                raise Exception("No interior support in linear facet form.")
            # initialize sparse matrix structures
            data = np.zeros(Nbfun_v*ne)
            rows = np.zeros(Nbfun_v*ne)
            cols = np.zeros(Nbfun_v*ne)

            for i in range(Nbfun_v):
                v1, dv1 = self.elem_v.gbasis(self.mapping, Y1, i, tind1)

                # find correct location in data,rows,cols
                ixs = slice(ne*i, ne*(i + 1))

                # compute entries of local stiffness matrices
                data[ixs] = np.dot(fform(v1, dv1, x, h, n, w, dw)*np.abs(detDG), W)
                rows[ixs] = self.dofnum_v.t_dof[i, tind1]
                cols[ixs] = np.zeros(ne)

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, 1)).toarray().T[0]

    def fnorm(self, form, interp, intorder=None, interior=False, normals=True):
        if interior:
            # evaluate norm on all interior facets
            find = self.mesh.interior_facets()
        else:
            # evaluate norm on all boundary facets
            find = self.mesh.boundary_facets()

        if not isinstance(interp, dict):
            raise Exception("The input solution vector(s) must be in a "
                            "dictionary! Pass e.g. {0:u} instead of u.")

        if intorder is None:
            intorder = 2*self.elem_u.maxdeg

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        if 'u1' in oldparams or 'du1' in oldparams:
            if interior is False:
                raise Exception("The supplied form contains u1 although "
                                "no interior=True is given.")
        if interior:
            paramlist = ['u1', 'u2', 'du1', 'du2',
                         'x', 'n', 't', 'h']
        else:
            paramlist = ['u', 'du', 'x', 'n', 't', 'h']
        fform = self.fillargs(form, paramlist)

        X, W = get_quadrature(self.mesh.brefdom, intorder)

        # indices of elements at different sides of facets
        tind1 = self.mesh.f2t[0, find]
        tind2 = self.mesh.f2t[1, find]

        # mappings
        x = self.mapping.G(X, find=find) # reference facet to global facet
        Y1 = self.mapping.invF(x, tind=tind1) # global facet to ref element
        Y2 = self.mapping.invF(x, tind=tind2) # global facet to ref element
        detDG = self.mapping.detDG(X, find)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        dim = self.mesh.p.shape[0]

        n = {}
        t = {}
        if normals:
            Y = self.mapping.invF(x, tind=tind1) # global facet to ref element
            n = self.mapping.normals(Y, tind1, find, self.mesh.t2f)
            if len(n) == 2: # TODO fix for 3D and other than triangles?
                t[0] = -n[1]
                t[1] = n[0]

        # interpolate some previous discrete function at quadrature points
        w1, w2 = {}, {}
        dw1, dw2 = {}, {}
        if interp is not None:
            if not isinstance(interp, dict):
                raise Exception("The input solution vector(s) must be in a "
                                "dictionary! Pass e.g. {0:u} instead of u.")
            for k in interp:
                w1[k] = 0.0*x[0]
                dw1[k] = const_cell(0.0*x[0], dim)
                w2[k] = 0.0*x[0]
                dw2[k] = const_cell(0.0*x[0], dim)
                for j in range(Nbfun_u):
                    phi1, dphi1 = self.elem_u.gbasis(self.mapping, Y1, j, tind1)
                    phi2, dphi2 = self.elem_u.gbasis(self.mapping, Y2, j, tind2)
                    w1[k] += interp[k][self.dofnum_u.t_dof[j, tind1], None]*phi1
                    w2[k] += interp[k][self.dofnum_u.t_dof[j, tind2], None]*phi2
                    for a in range(dim):
                        dw1[k][a] += interp[k][self.dofnum_u.t_dof[j, tind1], None]*dphi1[a]
                        dw2[k][a] += interp[k][self.dofnum_u.t_dof[j, tind2], None]*dphi2[a]


        h = np.abs(detDG)**(1.0/(self.mesh.dim()-1.0))

        if interior:
            return np.dot(fform(w1, w2, dw1, dw2,
                                x, n, t, h)**2*np.abs(detDG), W), find
        else:
            return np.dot(fform(w1, dw1,
                                x, n, t, h)**2*np.abs(detDG), W), find

    def inorm(self, form, interp, intorder=None):
        """Evaluate L2-norms of solution vectors inside elements. Useful for
        e.g. evaluating a posteriori estimators.

        Parameters
        ----------
        form : function handle
            The function for which the L2 norm is evaluated. Can consist
            of the following

            +-----------+----------------------+
            | Parameter | Explanation          |
            +-----------+----------------------+
            | u         | solution             |
            +-----------+----------------------+
            | du        | solution derivatives |
            +-----------+----------------------+
            | x         | spatial location     |
            +-----------+----------------------+
            | h         | the mesh parameter   |
            +-----------+----------------------+

            The function handle must use these exact names for
            the variables. Unused variable names can be omitted.

        interp : dict of numpy arrays
            The solutions that are interpolated.

        intorder : (OPTIONAL) int
            The order of polynomials for which the applied
            quadrature rule is exact. By default,
            2*Element.maxdeg is used.
        """
        # evaluate norm on all elements
        tind = range(self.mesh.t.shape[1])

        if not isinstance(interp, dict):
            raise Exception("The input solution vector(s) must be in a "
                            "dictionary! Pass e.g. {0:u} instead of u.")

        if intorder is None:
            intorder = 2*self.elem_u.maxdeg

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        paramlist = ['u', 'du', 'x', 'h']
        fform = self.fillargs(form, paramlist)

        X, W = get_quadrature(self.mesh.refdom, intorder)

        # mappings
        x = self.mapping.F(X, tind) # reference facet to global facet

        # jacobian at quadrature points
        detDF = self.mapping.detDF(X, tind)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        dim = self.mesh.p.shape[0]

        # interpolate the solution vectors at quadrature points
        zero = 0.0*x[0]
        w, dw = ({} for i in range(2))
        for k in interp:
            w[k] = zero
            dw[k] = const_cell(zero, dim)
            for j in range(Nbfun_u):
                jdofs = self.dofnum_u.t_dof[j, :]
                #phi, dphi = self.elem_u.lbasis(X, j)
                phi, dphi = self.elem_u.gbasis(self.mapping, X, j, tind)
                #w[k] += np.outer(interp[k][jdofs], phi)
                w[k] += interp[k][self.dofnum_u.t_dof[j, tind], None]*phi
                for a in range(dim):
                    #dw[k][a] += np.outer(interp[k][jdofs], dphi[a])
                    dw[k][a] += interp[k][self.dofnum_u.t_dof[j, tind], None]*dphi[a]

        # compute the mesh parameter from jacobian determinant
        h = np.abs(detDF)**(1.0/self.mesh.dim())

        return np.dot(fform(w, dw, x, h)**2*np.abs(detDF), W)

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
        self.n_dof = np.reshape(np.arange(element.n_dofs
                                          * mesh.p.shape[1],
                                          dtype=np.int64),
                                (element.n_dofs, mesh.p.shape[1]), order='F')
        offset = element.n_dofs*mesh.p.shape[1]

        # edge dofs
        if hasattr(mesh, 'edges'): # 3D mesh
            self.e_dof = np.reshape(np.arange(element.e_dofs
                                              * mesh.edges.shape[1],
                                              dtype=np.int64),
                                    (element.e_dofs, mesh.edges.shape[1]),
                                    order='F') + offset
            offset = offset + element.e_dofs*mesh.edges.shape[1]

        # facet dofs
        if hasattr(mesh, 'facets'): # 2D or 3D mesh
            self.f_dof = np.reshape(np.arange(element.f_dofs
                                              * mesh.facets.shape[1],
                                              dtype=np.int64),
                                    (element.f_dofs, mesh.facets.shape[1]),
                                    order='F') + offset
            offset = offset + element.f_dofs*mesh.facets.shape[1]

        # interior dofs
        self.i_dof = np.reshape(np.arange(element.i_dofs
                                          * mesh.t.shape[1],
                                          dtype=np.int64),
                                (element.i_dofs, mesh.t.shape[1]),
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

    def getdofs(self, N=None, F=None, E=None, T=None):
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
