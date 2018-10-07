import numpy as np
from skfem.assembly.dofs import Dofs

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

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

    def get_dofs(self, arg = None):
        """Return global DOF numbers.

        Parameters
        ----------
        arg
            The argument can have multiple meanings:
            
            - If an object of type :class:`~skfem.mesh.Submesh`, return corresponding Dofs object.
            - If a dictionary consisting of :class:`~skfem.mesh.Submesh` objects, return a dictionary
              of :class:`~skfem.mesh.Submesh` objects with same keys as the original dictionary.
            - If callable, first creates :class:`~skfem.mesh.Submesh` object through self.mesh.submesh(arg).
            - If None, first creates :class:`~skfem.mesh.Submesh` through self.mesh.submesh().

        """
        if arg is None:
            submesh = self.mesh.submesh()
        elif callable(arg):
            submesh = self.mesh.submesh(arg)
        else:
            submesh = arg
            
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
