import numpy as np
from skfem.assembly.dofs import Dofs

from typing import List, Optional, Tuple, Union, NamedTuple, Any

from numpy import ndarray


class GlobalBasis():
    """The finite element basis is evaluated at global quadrature points
    and cached inside this object.

    Please see the following implementations:

    - :class:`~skfem.assembly.InteriorBasis`, for basis functions inside elements
    - :class:`~skfem.assembly.FacetBasis`, for basis functions on element boundaries

    """

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

        self.nelems = None # subclasses should overwrite

        self.mesh = mesh

        self.refdom = mesh.refdom
        self.brefdom = mesh.brefdom

    def _build_dofnum(self, mesh, element):
        # vertex dofs
        self.nodal_dofs = np.reshape(
            np.arange(element.nodal_dofs * mesh.p.shape[1], dtype=np.int64),
            (element.nodal_dofs, mesh.p.shape[1]),
            order='F')
        offset = element.nodal_dofs * mesh.p.shape[1]

        # edge dofs
        if mesh.dim() == 3: 
            self.edge_dofs = np.reshape(
                np.arange(element.edge_dofs * mesh.edges.shape[1], dtype=np.int64),
                (element.edge_dofs, mesh.edges.shape[1]),
                order='F') + offset
            offset = offset + element.edge_dofs * mesh.edges.shape[1]

        # facet dofs
        if mesh.dim() >= 2: # 2D or 3D mesh
            self.facet_dofs = np.reshape(
                np.arange(element.facet_dofs * mesh.facets.shape[1], dtype=np.int64),
                (element.facet_dofs, mesh.facets.shape[1]),
                order='F') + offset
            offset = offset + element.facet_dofs * mesh.facets.shape[1]

        # interior dofs
        self.interior_dofs = np.reshape(
            np.arange(element.interior_dofs * mesh.t.shape[1], dtype=np.int64),
            (element.interior_dofs, mesh.t.shape[1]),
            order='F') + offset

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
        if type(D[0]) is dict:
            # if a dict of Dofs objects are given, flatten all
            D = tuple(D[0][key].all() for key in D[0])
        return np.setdiff1d(np.arange(self.N), np.concatenate(D))

    def _expand_facets(self, facets):
        """Transform a set of facets into facets, edges and points."""
        class IndexSet(NamedTuple):
            p: ndarray = None
            t: ndarray = None
            edges: ndarray = None
            facets: ndarray = None

        p = np.unique(self.mesh.facets[:, facets].flatten())

        if self.mesh.dim() == 3:
            edges = np.intersect1d(
                self.mesh.boundary_edges(),
                np.unique(self.mesh.t2e[:, self.mesh.f2t[0, facets]].flatten()))
            return IndexSet(p=p, edges=edges, facets=facets)
        else:
            return IndexSet(p=p, facets=facets)

    def _get_dofs(self, facets):
        """Return global DOF numbers corresponding to a set of facets."""
        ix = self._expand_facets(facets)
        nodal_dofs = {}
        facet_dofs = {}
        edge_dofs = {}
        interior_dofs = {}
        offset = 0

        if ix.p is not None:
            for i in range(self.nodal_dofs.shape[0]):
                nodal_dofs[self.dofnames[i]] = self.nodal_dofs[i, ix.p]
            offset += self.nodal_dofs.shape[0]
        if ix.facets is not None:
            for i in range(self.facet_dofs.shape[0]):
                facet_dofs[self.dofnames[i + offset]] = self.facet_dofs[i, ix.facets]
            offset += self.facet_dofs.shape[0]
        if ix.edges is not None:
            for i in range(self.edge_dofs.shape[0]):
                edge_dofs[self.dofnames[i + offset]] = self.edge_dofs[i, ix.edges]
            offset += self.edge_dofs.shape[0]
        if ix.t is not None:
            for i in range(self.interior_dofs.shape[0]):
                interior_dofs[self.dofnames[i + offset]] = self.interior_dofs[i, ix.t]

        return Dofs(nodal_dofs, facet_dofs, edge_dofs, interior_dofs)

    def get_dofs(self, facets = None):
        """Return global DOF numbers corresponding to facets (e.g. boundaries).

        Parameters
        ----------
        facets
            A list of facet indices. Alternatively:

            - if None, find facets by Mesh.boundary_facets()
            - if callable, call Mesh.facets_satisfying to get facets
            - if array, find the corresponding dofs
            - if dict of arrays, find dofs for each entry

        """
        if facets is None:
            facets = self.mesh.boundary_facets()
        elif callable(facets):
            facets = self.mesh.facets_satisfying(facets)

        if type(facets) is dict:
            return {key: self._get_dofs(facets[key]) for key in facets}
        else:
            return self._get_dofs(facets)

    def default_parameters(self):
        """This is used by :func:`skfem.assembly.asm` to get the default
        parameters for 'w'."""
        raise NotImplementedError("Default parameters not implemented.")

    def interpolate(self,
                    w: ndarray) -> Any:
        """Interpolate a solution vector to quadrature points.

        Parameters
        ----------
        w
            A solution vector.

        Returns
        -------
        ndarray
            Interpolated solution vector.
        ndarray
            Interpolated derivatives of the solution vector.

        """
        nqp = len(self.W)
        dim = self.mesh.dim()

        if self.elem.order[0] == 0:
            W = np.zeros((self.nelems, nqp))
        elif self.elem.order[0] == 1:
            W = np.zeros((dim, self.nelems, nqp))
        else:
            raise Exception("Interpolation not implemented for this Element.order.")

        for j in range(self.Nbfun):
            jdofs = self.element_dofs[j, :]
            W += w[jdofs][:, None] \
                 * self.basis[j][0]

        if self.elem.order[1] == 1:
            dW = np.zeros((dim, self.nelems, nqp))
        elif self.elem.order[1] == 2:
            dW = np.zeros((dim, dim, self.nelems, nqp))
        else:
            raise Exception("Interpolation not implemented for this Element.order.")
        for j in range(self.Nbfun):
            jdofs = self.element_dofs[j, :]
            for a in range(dim):
                dW[a, :, :] += w[jdofs][:, None] \
                               * self.basis[j][1][a]
        return W, dW

    def zero_w(self) -> ndarray:
        """Return a zero array with correct dimensions
        for :func:`~skfem.assembly.asm`."""
        return np.zeros((self.nelems, len(self.W)))
