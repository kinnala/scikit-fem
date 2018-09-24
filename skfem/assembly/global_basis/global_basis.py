from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy import ndarray


class GlobalBasis():
    """The finite element basis is evaluated at global quadrature points
    and cached inside this object.

    Please see the following implementations:

    - :class:`~skfem.assembly.InteriorBasis`, for basis functions inside elements
    - :class:`~skfem.assembly.FacetBasis`, for basis functions on element boundaries

    """

    nodal_dofs: ndarray = np.array([])
    facet_dofs: ndarray = np.array([])
    edge_dofs: ndarray = np.array([])
    interior_dofs: ndarray = np.array([])
    element_dofs: ndarray = np.array([])
    N: int = 0
    dofnames: List[str] = []

    def __init__(self, mesh, elem, mapping, intorder):
        if mapping is None:
            self.mapping = mesh.mapping()
        else:
            self.mapping = mapping

        self._build_dofnum(mesh, elem)
        self.elem = elem
        self.Nbfun = self.element_dofs.shape[0]

        if intorder is None:
            self.intorder = 2*self.elem.maxdeg
        else:
            self.intorder = intorder

        self.dim = mesh.p.shape[0]
        self.nt = mesh.t.shape[1]

        self.mesh = mesh

        self.refdom = mesh.refdom
        self.brefdom = mesh.brefdom

    def _build_dofnum(self, mesh, element):
        # vertex dofs
        self.nodal_dofs = np.reshape(np.arange(element.nodal_dofs * mesh.p.shape[1], dtype=np.int64),
                                     (element.nodal_dofs, mesh.p.shape[1]), order='F')
        offset = element.nodal_dofs * mesh.p.shape[1]

        # edge dofs
        if mesh.dim() == 3: 
            self.edge_dofs = np.reshape(np.arange(element.edge_dofs * mesh.edges.shape[1], dtype=np.int64),
                                        (element.edge_dofs, mesh.edges.shape[1]), order='F') + offset
            offset = offset + element.edge_dofs * mesh.edges.shape[1]

        # facet dofs
        if mesh.dim() >= 2: # 2D or 3D mesh
            self.facet_dofs = np.reshape(np.arange(element.facet_dofs * mesh.facets.shape[1], dtype=np.int64),
                                         (element.facet_dofs, mesh.facets.shape[1]), order='F') + offset
            offset = offset + element.facet_dofs * mesh.facets.shape[1]

        # interior dofs
        self.interior_dofs = np.reshape(np.arange(element.interior_dofs * mesh.t.shape[1], dtype=np.int64),
                                        (element.interior_dofs, mesh.t.shape[1]), order='F') + offset

        # global numbering
        self.element_dofs = np.zeros((0, mesh.t.shape[1]), dtype=np.int64)

        # nodal dofs
        for itr in range(mesh.t.shape[0]):
            self.element_dofs = np.vstack((self.element_dofs,
                                           self.nodal_dofs[:, mesh.t[itr, :]]))

        # edge dofs
        if mesh.dim() == 3:
            for itr in range(mesh.t2e.shape[0]):
                self.element_dofs = np.vstack((self.element_dofs,
                                               self.edge_dofs[:, mesh.t2e[itr, :]]))

        # facet dofs
        if mesh.dim() >= 2:
            for itr in range(mesh.t2f.shape[0]):
                self.element_dofs = np.vstack((self.element_dofs,
                                               self.facet_dofs[:, mesh.t2f[itr, :]]))

        self.element_dofs = np.vstack((self.element_dofs, self.interior_dofs))

        self.N = np.max(self.element_dofs) + 1
        self.dofnames = element.dofnames

    def complement_dofs(self, *D):
        return np.setdiff1d(np.arange(self.N), np.concatenate(D))

    def _get_dofs(self, submesh):
        """Return global DOF numbers corresponding to a Submesh."""
        class Dofs(NamedTuple):
            nodal: Dict[str, ndarray] = {}
            facet: Dict[str, ndarray] = {}
            edge: Dict[str, ndarray] = {}
            interior: Dict[str, ndarray] = {}

        nodal_dofs = {}
        facet_dofs = {}
        edge_dofs = {}
        interior_dofs = {}
        offset = 0

        if submesh.p is not None:
            for i in range(self.nodal_dofs.shape[0]):
                nodal_dofs[self.dofnames[i]] = self.nodal_dofs[i, submesh.p]
            offset += self.nodal_dofs.shape[0]
        if submesh.facets is not None:
            for i in range(self.facet_dofs.shape[0]):
                facet_dofs[self.dofnames[i + offset]] = self.facet_dofs[i, submesh.facets]
            offset += self.facet_dofs.shape[0]
        if submesh.edges is not None:
            for i in range(self.edge_dofs.shape[0]):
                edge_dofs[self.dofnames[i + offset]] = self.edge_dofs[i, submesh.edges]
            offset += self.edge_dofs.shape[0]
        if submesh.t is not None:
            for i in range(self.interior_dofs.shape[0]):
                interior_dofs[self.dofnames[i + offset]] = self.interior_dofs[i, submesh.t]

        return Dofs(nodal_dofs, facet_dofs, edge_dofs, interior_dofs)

    def get_dofs(self, submesh):
        """Return global DOF numbers corresponding to one or multiple
        Submeshes."""
        class Dofs(NamedTuple):
            nodal: Dict[str, ndarray] = {}
            facet: Dict[str, ndarray] = {}
            edge: Dict[str, ndarray] = {}
            interior: Dict[str, ndarray] = {}
        if type(submesh) is dict:
            return {key: self._get_dofs(submesh[key]) for key in submesh}
        else:
            return self._get_dofs(submesh)

    def init_gbasis(self, nvals, nqp, order):
        if order == 0:
            return np.empty((self.Nbfun, nvals, nqp))
        else:
            return np.empty((self.Nbfun,) + order*(self.dim,) + (nvals, nqp))

    def default_parameters(self):
        """This is used by :func:`skfem.assembly.asm` to get the default
        parameters for 'w'."""
        raise NotImplementedError("Default parameters not implemented.")

    def interpolate(self,
                    w: ndarray,
                    derivative: Optional[bool] = False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """Interpolate a solution vector to quadrature points.

        Parameters
        ----------
        w
            A solution vector.
        derivative
            If true, return also the derivative.

        Returns
        -------
        ndarray
            Interpolated solution vector (Nelems x Nqp).

        """
        nqp = len(self.W)

        if self.elem.order[0] == 0:
            W = np.zeros((self.nelems, nqp))
        elif self.elem.order[0] == 1:
            W = np.zeros((self.dim, self.nelems, nqp))
        else:
            raise Exception("Interpolation not implemented for this Element.order.")

        for j in range(self.Nbfun):
            jdofs = self.element_dofs[j, :]
            W += w[jdofs][:, None] \
                 * self.basis[0][j]

        if derivative:
            if self.elem.order[1] == 1:
                dW = np.zeros((self.dim, self.nelems, nqp))
            elif self.elem.order[1] == 2:
                dW = np.zeros((self.dim, self.dim, self.nelems, nqp))
            else:
                raise Exception("Interpolation not implemented for this Element.order.")
            for j in range(self.Nbfun):
                jdofs = self.element_dofs[j, :]
                for a in range(self.dim):
                    dW[a, :, :] += w[jdofs][:, None] \
                                   * self.basis[1][j][a]
            return W, dW
        return W

    def find_dofs(self, test=None, bc=None, boundary=True, dofrows=None,
                  check_vertices=True, check_facets=True, check_edges=True):
        """Helper function for finding DOF indices for BC's.

        Does not test for element interior DOFs since they are not typically
        included in boundary conditions! Uses DOF numbering of 'u' variable.

        Parameters
        ----------
        test : (optional, default=function returning True) lambda
            An anonymous function with Ndim arguments. If returns other than 0
            when evaluated at the DOF location, the respective DOF is included
            in the return set.
        bc : (optional, default=zero function) lambda
            The boundary condition value.
        boundary : (optional, default=True) bool
            Check only boundary DOFs.
        dofrows : (optional, default=None) np.array
            List of rows that are extracted from the DOF structures.
            For example, if each node/facet/edge contains 3 DOFs (say, in three
            dimensional problems x, y and z displacements) you can give [0, 1]
            to consider only two first DOFs.
        check_vertices : (optional, default=True) bool
            Include vertex dofs
        check_facets: (optional, default=True) bool
            Include facet dofs
        check_edges: (optional, default=True) bool
            Include edge dofs (3D only)

        Returns
        -------
        x : np.array
            Solution vector with the BC's
        I : np.array
            Set of DOF numbers set by the function

        """
        if test is None:
            if self.mesh.dim() == 1:
                test = lambda x: 0*x + True
            elif self.mesh.dim() == 2:
                test = lambda x, y: 0*x + True
            elif self.mesh.dim() == 3:
                test = lambda x, y, z: 0*x + True

        if bc is None:
            if self.mesh.dim() == 1:
                bc = lambda x: 0*x
            elif self.mesh.dim() == 2:
                bc = lambda x, y: 0*x
            elif self.mesh.dim() == 3:
                bc = lambda x, y, z: 0*x

        x = np.zeros(self.N)

        dofs = np.zeros(0, dtype=np.int64)
        locs = np.zeros((self.mesh.dim(), 0))

        if check_vertices:
            # handle nodes
            N = self.mesh.nodes_satisfying(test)
            if boundary:
                N = np.intersect1d(N, self.mesh.boundary_nodes())
            if dofrows is None:
                Ndofs = self.nodal_dofs[:, N]
            else:
                Ndofs = self.nodal_dofs[dofrows][:, N]

            Ndofx = np.tile(self.mesh.p[0, N], (Ndofs.shape[0], 1)).flatten()
            Ndofy = np.tile(self.mesh.p[1, N], (Ndofs.shape[0], 1)).flatten()
            if self.mesh.dim() == 3:
                Ndofz = np.tile(self.mesh.p[2, N], (Ndofs.shape[0], 1)).flatten()
                locs = np.hstack((locs, np.vstack((Ndofx, Ndofy, Ndofz))))
            else:
                locs = np.hstack((locs, np.vstack((Ndofx, Ndofy))))

            dofs = np.hstack((dofs, Ndofs.flatten()))

        if check_facets and self.facet_dofs.shape[0]>0:
            # handle facets
            F = self.mesh.facets_satisfying(test)
            if boundary:
                F = np.intersect1d(F, self.mesh.boundary_facets())
            if dofrows is None:
                Fdofs = self.facet_dofs[:, F]
            else:
                Fdofs = self.facet_dofs[dofrows][:, F]

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

        if check_edges and self.edge_dofs.shape[0]>0:
            # handle edges
            if self.mesh.dim() == 3:
                E = self.mesh.edges_satisfying(test)
                if boundary:
                    E = np.intersect1d(E, self.mesh.boundary_edges())
                if dofrows is None:
                    Edofs = self.edge_dofs[:, E]
                else:
                    Edofs = self.edge_dofs[dofrows][:, E]

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
