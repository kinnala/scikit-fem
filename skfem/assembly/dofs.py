from typing import Union, NamedTuple, Any, List, Optional

import numpy as np
from numpy import ndarray

from skfem.element import Element
from skfem.mesh import Mesh


class DofsView(NamedTuple):
    """A subset of :class:`skfem.assembly.Dofs`."""

    obj: Any = None
    nodal_ix: Union[ndarray, slice] = slice(None)
    facet_ix: Union[ndarray, slice] = slice(None)
    edge_ix: Union[ndarray, slice] = slice(None)
    interior_ix: Union[ndarray, slice] = slice(None)
    nodal_rows: Union[ndarray, slice] = slice(None)
    facet_rows: Union[ndarray, slice] = slice(None)
    edge_rows: Union[ndarray, slice] = slice(None)
    interior_rows: Union[ndarray, slice] = slice(None)

    def flatten(self) -> ndarray:
        """Return all DOF indices as a single array."""
        return np.unique(
            np.concatenate((
                (self.obj
                 .nodal_dofs[self.nodal_rows][:, self.nodal_ix]
                 .flatten()),
                (self.obj
                 .facet_dofs[self.facet_rows][:, self.facet_ix]
                 .flatten()),
                (self.obj
                 .edge_dofs[self.edge_rows][:, self.edge_ix]
                 .flatten()),
                (self.obj
                 .interior_dofs[self.interior_rows][:, self.interior_ix]
                 .flatten())
            ))
        )

    def _intersect(self, a, b):
        if isinstance(a, slice):
            if a.start == 0:
                return a
            return b
        if isinstance(b, slice):
            if b.start == 0:
                return b
            return a
        return np.intersect1d(a, b)

    def _intersect_tuples(self, a, b):
        return tuple([self._intersect(a[i], b[i]) for i in range(len(a))])

    def keep(self, dofnames: List[str]):
        """Keep DOFs with the given names and remove others.

        Parameters
        ----------
        dofnames
            An array of DOF names, e.g. `["u", "u_n"]`.

        """
        return DofsView(
            self.obj,
            self.nodal_ix,
            self.facet_ix,
            self.edge_ix,
            self.interior_ix,
            *self._intersect_tuples(
                (self.nodal_rows,
                 self.facet_rows,
                 self.edge_rows,
                 self.interior_rows),
                self._dofnames_to_rows(dofnames)
            )
        )

    def drop(self, dofnames):
        """Remove DOFs with the given names.

        Parameters
        ----------
        dofnames
            An array of DOF names, e.g. `["u", "u_n"]`.

        """
        return DofsView(
            self.obj,
            self.nodal_ix,
            self.facet_ix,
            self.edge_ix,
            self.interior_ix,
            *self._intersect_tuples(
                (self.nodal_rows,
                 self.facet_rows,
                 self.edge_rows,
                 self.interior_rows),
                self._dofnames_to_rows(dofnames, skip=True)
            )
        )

    def all(self, key=None):
        if key is None:
            return self.flatten()
        return self.keep(key).flatten()

    def __array__(self):
        return self.flatten()

    @property
    def nodal(self):
        return self._by_name(self.nodal_dofs[self.nodal_rows],
                             ix=self.nodal_ix,
                             rows=self.nodal_rows)

    @property
    def facet(self):
        return self._by_name(self.facet_dofs[self.facet_rows],
                             off=self.nodal_dofs.shape[0],
                             ix=self.facet_ix,
                             rows=self.facet_rows)

    @property
    def edge(self):
        return self._by_name(self.edge_dofs[self.edge_rows],
                             off=(self.nodal_dofs.shape[0]
                                  + self.facet_dofs.shape[0]),
                             ix=self.edge_ix,
                             rows=self.edge_rows)

    @property
    def interior(self):
        return self._by_name(self.interior_dofs[self.interior_rows],
                             off=(self.nodal_dofs.shape[0]
                                  + self.facet_dofs.shape[0]
                                  + self.edge_dofs.shape[0]),
                             ix=self.interior_ix,
                             rows=self.interior_rows)

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    def __or__(self, other):
        """For merging two sets of DOFs."""
        return DofsView(
            self.obj,
            np.union1d(self.nodal_ix, other.nodal_ix),
            np.union1d(self.facet_ix, other.facet_ix),
            np.union1d(self.edge_ix, other.edge_ix),
            np.union1d(self.interior_ix, other.interior_ix)
        )

    def __add__(self, other):
        return self.__or__(other)


class Dofs:
    """An object containing a set of degree-of-freedom indices."""

    nodal_dofs: Optional[ndarray] = None
    facet_dofs: Optional[ndarray] = None
    edge_dofs: Optional[ndarray] = None
    interior_dofs: Optional[ndarray] = None

    element_dofs: Optional[ndarray] = None
    N: int = 0

    topo: Mesh
    element: Element

    def __init__(self, topo, element):

        self.topo = topo
        self.element = element

        self.nodal_dofs = np.reshape(
            np.arange(element.nodal_dofs * topo.nvertices, dtype=np.int64),
            (element.nodal_dofs, topo.nvertices),
            order='F')
        offset = element.nodal_dofs * topo.nvertices

        # edge dofs
        if element.dim == 3 and element.edge_dofs > 0:
            self.edge_dofs = np.reshape(
                np.arange(element.edge_dofs * topo.nedges,
                          dtype=np.int64),
                (element.edge_dofs, topo.nedges),
                order='F') + offset
            offset += element.edge_dofs * topo.nedges
        else:
            self.edge_dofs = np.empty((0, 0), dtype=np.int64)

        # facet dofs
        if element.facet_dofs > 0:
            self.facet_dofs = np.reshape(
                np.arange(element.facet_dofs * topo.nfacets,
                          dtype=np.int64),
                (element.facet_dofs, topo.nfacets),
                order='F') + offset
            offset += element.facet_dofs * topo.nfacets
        else:
            self.facet_dofs = np.empty((0, 0), dtype=np.int64)

        # interior dofs
        self.interior_dofs = np.reshape(
            np.arange(element.interior_dofs * topo.nelements, dtype=np.int64),
            (element.interior_dofs, topo.nelements),
            order='F') + offset

        # global numbering
        self.element_dofs = np.zeros((0, topo.nelements), dtype=np.int64)

        # nodal dofs
        for itr in range(topo.t.shape[0]):
            self.element_dofs = np.vstack((
                self.element_dofs,
                self.nodal_dofs[:, topo.t[itr]]
            ))

        # edge dofs
        if element.dim == 3 and element.edge_dofs > 0:
            for itr in range(topo.t2e.shape[0]):
                self.element_dofs = np.vstack((
                    self.element_dofs,
                    self.edge_dofs[:, topo.t2e[itr]]
                ))

        # facet dofs
        if element.dim >= 2 and element.facet_dofs > 0:
            for itr in range(topo.t2f.shape[0]):
                self.element_dofs = np.vstack((
                    self.element_dofs,
                    self.facet_dofs[:, topo.t2f[itr]]
                ))

        # interior dofs
        self.element_dofs = np.vstack((self.element_dofs,
                                       self.interior_dofs))

        # total dofs
        self.N = np.max(self.element_dofs) + 1

    def get_facet_dofs(self,
                       facets: ndarray,
                       skip_dofnames: List[str] = None) -> DofsView:
        """Return a subset of DOFs corresponding to the given facets.

        Parameters
        ----------
        facets
            An array of facet indices.
        skip_dofnames
            An array of dofnames to skip.

        """
        if self.element.nodal_dofs > 0 or self.element.edge_dofs > 0:
            nodal_ix, edge_ix = self.topo._expand_facets(facets)

        nodal_ix = (np.empty((0,), dtype=np.int64)
                    if self.element.nodal_dofs == 0
                    else nodal_ix)
        edge_ix = (np.empty((0,), dtype=np.int64)
                   if self.element.edge_dofs == 0
                   else edge_ix)
        facet_ix = (np.empty((0,), dtype=np.int64)
                    if self.element.facet_dofs == 0
                    else facets)

        if skip_dofnames is None:
            skip_dofnames = []

        return DofsView(
            self,
            nodal_ix,
            facet_ix,
            edge_ix,
            np.empty((0,), dtype=np.int64),
            *self._dofnames_to_rows(skip_dofnames, skip=True)
        )

    def _by_name(self,
                 dofs: ndarray,
                 off: int = 0,
                 ix: Optional[ndarray] = None,
                 rows: Optional[Union[List[int], ndarray]] = None):

        n_dofs = dofs.shape[0]
        n_ents = dofs.shape[1] if ix is None else len(ix)

        if rows is None:
            rows = list(range(n_dofs))

        ents = {
            self.element.dofnames[rows[i] + off]: np.zeros((0, n_ents),
                                                           dtype=np.int64)
            for i in range(n_dofs)
        }
        for i in range(n_dofs):
            new_row = dofs[i] if ix is None else dofs[i, ix]
            ents[self.element.dofnames[rows[i] + off]] =\
                np.vstack((ents[self.element.dofnames[rows[i] + off]],
                           new_row))

        return {k: ents[k].flatten() for k in ents}

    def _dofnames_to_rows(self, dofnames, skip=False):

        if isinstance(dofnames, str):
            dofnames = [dofnames]

        if skip is True:
            def check(x, y):
                return x not in y
        else:
            def check(x, y):
                return x in y

        n_nodal = self.nodal_dofs.shape[0]
        n_facet = self.facet_dofs.shape[0]
        n_edge = self.edge_dofs.shape[0]
        n_interior = self.interior_dofs.shape[0]

        nodal_rows = []
        for i in range(n_nodal):
            if check(self.element.dofnames[i], dofnames):
                nodal_rows.append(i)

        facet_rows = []
        for i in range(n_facet):
            if check(self.element.dofnames[i + n_nodal], dofnames):
                facet_rows.append(i)

        edge_rows = []
        for i in range(n_edge):
            if check(self.element.dofnames[i + n_nodal + n_facet], dofnames):
                edge_rows.append(i)

        interior_rows = []
        for i in range(n_interior):
            if check(self.element.dofnames[i + n_nodal + n_facet + n_edge],
                     dofnames):
                interior_rows.append(i)

        return (
            nodal_rows if len(nodal_rows) > 0 else slice(0, 0),
            facet_rows if len(facet_rows) > 0 else slice(0, 0),
            edge_rows if len(edge_rows) > 0 else slice(0, 0),
            interior_rows if len(interior_rows) > 0 else slice(0, 0)
        )
