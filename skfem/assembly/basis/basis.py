import warnings
from typing import List, Optional, NamedTuple, Any

import numpy as np
from numpy import ndarray

from skfem.assembly.dofs import Dofs
from skfem.element.element import DiscreteField
from skfem.element.element_composite import ElementComposite


class Basis():
    """The finite element basis is evaluated at global quadrature points
    and cached inside this object.

    Please see the following implementations:

    - :class:`~skfem.assembly.InteriorBasis`, basis functions inside elements
    - :class:`~skfem.assembly.FacetBasis`, basis functions on element boundaries
    - :class:`~skfem.assembly.MortarBasis`, basis functions on mortar interfaces

    """

    N: int = 0
    dofnames: List[str] = []

    def __init__(self, mesh, elem, mapping, intorder):
        if mapping is None:
            self.mapping = mesh.mapping()
        else:
            self.mapping = mapping

        self._build_dofnum(mesh, elem)

        # human readable names
        self.dofnames = elem.dofnames

        # global degree-of-freedom location
        if hasattr(elem, 'doflocs'):
            doflocs = self.mapping.F(elem.doflocs.T)
            self.doflocs = np.zeros((doflocs.shape[0], self.N))
            for itr in range(doflocs.shape[0]):
                for jtr in range(self.element_dofs.shape[0]):
                    self.doflocs[itr, self.element_dofs[jtr]] =\
                        doflocs[itr, :, jtr]

        self.mesh = mesh
        self.elem = elem

        self.Nbfun = self.element_dofs.shape[0]

        if intorder is None:
            self.intorder = 2 * self.elem.maxdeg
        else:
            self.intorder = intorder

        self.nelems = None # subclasses should overwrite

        self.refdom = mesh.refdom
        self.brefdom = mesh.brefdom

    def _build_dofnum(self, mesh, element):
        """Build global degree-of-freedom numbering."""
        # vertex dofs
        self.nodal_dofs = np.reshape(
            np.arange(element.nodal_dofs * mesh.p.shape[1], dtype=np.int64),
            (element.nodal_dofs, mesh.p.shape[1]),
            order='F')
        offset = element.nodal_dofs * mesh.p.shape[1]

        # edge dofs
        if mesh.dim() == 3: 
            self.edge_dofs = np.reshape(
                np.arange(element.edge_dofs * mesh.edges.shape[1],
                          dtype=np.int64),
                (element.edge_dofs, mesh.edges.shape[1]),
                order='F') + offset
            offset = offset + element.edge_dofs * mesh.edges.shape[1]
        else:
            self.edge_dofs = np.empty((0,0))

        # facet dofs
        self.facet_dofs = np.reshape(
            np.arange(element.facet_dofs * mesh.facets.shape[1],
                      dtype=np.int64),
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
            self.element_dofs = np.vstack((
                self.element_dofs,
                self.nodal_dofs[:, mesh.t[itr]]
            ))

        # edge dofs
        if mesh.dim() == 3:
            for itr in range(mesh.t2e.shape[0]):
                self.element_dofs = np.vstack((
                    self.element_dofs,
                    self.edge_dofs[:, mesh.t2e[itr]]
                ))

        # facet dofs
        if mesh.dim() >= 2:
            for itr in range(mesh.t2f.shape[0]):
                self.element_dofs = np.vstack((
                    self.element_dofs,
                    self.facet_dofs[:, mesh.t2f[itr]]
                ))

        # interior dofs
        self.element_dofs = np.vstack((self.element_dofs, self.interior_dofs))

        # total dofs
        self.N = np.max(self.element_dofs) + 1

    def complement_dofs(self, *D):
        if type(D[0]) is dict:
            # if a dict of Dofs objects are given, flatten all
            D = tuple(D[0][key].all() for key in D[0])
        return np.setdiff1d(np.arange(self.N), np.concatenate(D))

    def _get_dofs(self, facets):
        """Return :class:`skfem.assembly.Dofs` corresponding to a set of facets."""
        nodal_ix = np.unique(self.mesh.facets[:, facets].flatten())
        facet_ix = facets
        if self.mesh.dim() == 3:
            edge_ix = np.intersect1d(
                self.mesh.boundary_edges(),
                np.unique(self.mesh.t2e[:, self.mesh.f2t[0, facets]].flatten()))
        else:
            edge_ix = []

        n_nodal = self.nodal_dofs.shape[0]
        n_facet = self.facet_dofs.shape[0]
        n_edge = self.edge_dofs.shape[0]

        return Dofs(
            nodal = {self.dofnames[i]: self.nodal_dofs[i, nodal_ix]
                     for i in range(n_nodal)},
            facet = {self.dofnames[i + n_nodal]: self.facet_dofs[i, facet_ix]
                     for i in range(n_facet)},
            edge = {self.dofnames[i + n_nodal + n_facet]: self.edge_dofs[i, edge_ix]
                     for i in range(n_edge)},
        )

    def get_dofs(self, facets: Optional[Any] = None):
        """Return global DOF numbers corresponding to facets (e.g. boundaries).

        Parameters
        ----------
        facets
            A list of facet indices. If None, find facets by
            Mesh.boundary_facets().  If callable, call Mesh.facets_satisfying to
            get facets. If array, find the corresponding dofs. If dict of
            arrays, find dofs for each entry. If dict of callables, call
            Mesh.facets_satisfying for each entry to get facets and then find
            dofs for those.

        Returns
        -------
        Dofs
            A subset of degrees-of-freedom as :class:`skfem.assembly.dofs.Dofs`.


        """
        if facets is None:
            facets = self.mesh.boundary_facets()
        elif callable(facets):
            facets = self.mesh.facets_satisfying(facets)
        if isinstance(facets, dict):
            def to_indices(f):
                if callable(f):
                    return self.mesh.facets_satisfying(f)
                return f
            return {k: self._get_dofs(to_indices(facets[k])) for k in facets}
        else:
            return self._get_dofs(facets)

    def default_parameters(self):
        """This is used by :func:`skfem.assembly.asm` to get the default
        parameters for 'w'."""
        raise NotImplementedError("Default parameters not implemented.")

    def interpolate(self, w: ndarray) -> Any:
        """Interpolate a solution vector to quadrature points.

        Parameters
        ----------
        w
            A solution vector.

        Returns
        -------
        DiscreteField
            The solution vector interpolated at quadrature points.

        """
        if w.shape[0] != self.N:
            raise ValueError("Input array has wrong size.")

        ref = self.basis[0][0]
        field = []

        def linear_combination(n): # TODO make work with composite elements
            out = 0. * ref[n].copy()
            for i in range(self.Nbfun):
                values = w[self.element_dofs[i]][:, None]
                if len(ref[n].shape) == 2:
                    out += values * self.basis[i][0][n]
                elif len(ref[n].shape) == 3:
                    for j in range(out.shape[0]):
                        out[j, :, :] += values * self.basis[i][0][n][j]
                elif len(ref[n].shape) == 4:
                    for j in range(out.shape[0]):
                        for k in range(out.shape[1]):
                            out[j, k, :, :] += \
                                values * self.basis[i][0][n][j, k]
            return out

        for n in range(len(ref)):
            if ref[n] is not None:
                field.append(linear_combination(n))
            else:
                field.append(None)

        return DiscreteField(*field)

    def split(self):
        """Return indices to different solution components."""
        if isinstance(self.elem, ElementComposite):
            off1 = 0
            off2 = 0
            off3 = 0
            off4 = 0
            output = [None] * len(self.elem.elems)
            for k in range(len(self.elem.elems)):
                #output[k] = np.concatenate((
                e = self.elem.elems[k]
                n1 = e.nodal_dofs
                n2 = e.edge_dofs
                n3 = e.facet_dofs
                n4 = e.interior_dofs
                output[k] = np.concatenate((
                    self.nodal_dofs[off1:(off1 + n1)].flatten(),
                    self.edge_dofs[off2:(off2 + n2)].flatten(),
                    self.facet_dofs[off3:(off3 + n3)].flatten(),
                    self.interior_dofs[off4:(off4 + n4)].flatten(),
                )).astype(np.int)
                off1 += n1
                off2 += n2
                off3 += n3
                off4 += n4
            return tuple(output)
        raise ValueError("Basis.elem has only single component!")

    def zero_w(self) -> ndarray:
        """Return a zero array with correct dimensions
        for :func:`~skfem.assembly.asm`."""
        return np.zeros((self.nelems, len(self.W)))
