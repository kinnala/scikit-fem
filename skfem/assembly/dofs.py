from typing import Union

import numpy as np
from numpy import ndarray

from skfem.element import Element
from skfem.mesh import Mesh


class DofsView:
    """A subset of :class:`skfem.assembly.Dofs`."""

    _nodal_ix: Union[ndarray, slice] = None
    _facet_ix: Union[ndarray, slice] = None
    _edge_ix: Union[ndarray, slice] = None
    _interior_ix: Union[ndarray, slice] = None

    _nodal_rows: Union[ndarray, slice] = None
    _facet_rows: Union[ndarray, slice] = None
    _edge_rows: Union[ndarray, slice] = None
    _interior_rows: Union[ndarray, slice] = None

    def __init__(self,
                 obj,
                 nodal_ix=None,
                 facet_ix=None,
                 edge_ix=None,
                 interior_ix=None,
                 nodal_rows=None,
                 facet_rows=None,
                 edge_rows=None,
                 interior_rows=None):
        self._nodal_ix = slice(None)\
            if nodal_ix is None else nodal_ix
        self._facet_ix = slice(None)\
            if facet_ix is None else facet_ix
        self._edge_ix = slice(None)\
            if edge_ix is None else edge_ix
        self._interior_ix = slice(None)\
            if interior_ix is None else interior_ix
        self._nodal_rows = slice(None)\
            if nodal_rows is None else nodal_rows
        self._facet_rows = slice(None)\
            if facet_rows is None else facet_rows
        self._edge_rows = slice(None)\
            if edge_rows is None else edge_rows
        self._interior_rows = slice(None)\
            if interior_rows is None else interior_rows
        self._obj = obj

    def flatten(self):
        return np.unique(
            np.concatenate((
                (self._obj
                 ._nodal_dofs[self._nodal_rows][:, self._nodal_ix]
                 .flatten()),
                (self._obj
                 ._facet_dofs[self._facet_rows][:, self._facet_ix]
                 .flatten()),
                (self._obj
                 ._edge_dofs[self._edge_rows][:, self._edge_ix]
                 .flatten()),
                (self._obj
                 ._interior_dofs[self._interior_rows][:, self._interior_ix]
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

    def keep(self, dofnames):
        return DofsView(
            self._obj,
            self._nodal_ix,
            self._facet_ix,
            self._edge_ix,
            self._interior_ix,
            *self._intersect_tuples(
                (self._nodal_rows,
                 self._facet_rows,
                 self._edge_rows,
                 self._interior_rows),
                self._dofnames_to_rows(dofnames)
            )
        )

    def skip(self, dofnames):
        return DofsView(
            self._obj,
            self._nodal_ix,
            self._facet_ix,
            self._edge_ix,
            self._interior_ix,
            *self._intersect_tuples(
                (self._nodal_rows,
                 self._facet_rows,
                 self._edge_rows,
                 self._interior_rows),
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
        return self._by_name(self._nodal_dofs,
                             ix=self._nodal_ix)

    @property
    def facet(self):
        return self._by_name(self._facet_dofs,
                             off=self._nodal_dofs.shape[0],
                             ix=self._facet_ix)

    @property
    def edge(self):
        return self._by_name(self._edge_dofs,
                             off=(self._nodal_dofs.shape[0]
                                  + self._facet_dofs.shape[0]),
                             ix=self._edge_ix)

    @property
    def interior(self):
        return self._by_name(self._interior_dofs,
                             off=(self._nodal_dofs.shape[0]
                                  + self._facet_dofs.shape[0]
                                  + self._edge_dofs.shape[0]),
                             ix=self._interior_ix)

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __or__(self, other):
        """For merging two sets of DOF's."""
        return DofsView(
            self._obj,
            np.union1d(self._nodal_ix, other._nodal_ix),
            np.union1d(self._facet_ix, other._facet_ix),
            np.union1d(self._edge_ix, other._edge_ix),
            np.union1d(self._interior_ix, other._interior_ix)
        )

    def __add__(self, other):
        return self.__or__(other)


class Dofs:
    """An object containing a set of degree-of-freedom indices."""

    _nodal_dofs: ndarray = None
    _facet_dofs: ndarray = None
    _edge_dofs: ndarray = None
    _interior_dofs: ndarray = None

    _element_dofs: ndarray = None
    N: int = 0

    topo: Mesh = None
    element: Element = None

    def _by_name(self,
                 dofs,
                 off=0,
                 ix=None,
                 rows=None):

        n_dofs = dofs.shape[0]

        if ix is None:
            n_ents = dofs.shape[1]
        else:
            n_ents = len(ix)

        ents = {
            self.element.dofnames[i + off]: np.zeros((0, n_ents),
                                                     dtype=np.int64)
            for i in range(n_dofs)
        }
        for i in range(n_dofs):
            if ix is None:
                ents[self.element.dofnames[i + off]] =\
                    np.vstack((ents[self.element.dofnames[i + off]],
                               dofs[i]))
            else:
                ents[self.element.dofnames[i + off]] =\
                    np.vstack((ents[self.element.dofnames[i + off]],
                               dofs[i, ix]))

        return {k: ents[k].flatten() for k in ents}

    def get_facet_dofs(self, facets: ndarray, skip_dofnames=None) -> DofsView:
        """Return a subset of DOF's corresponding to the given facets.

        Parameters
        ----------
        facets
            An array of facet indices.

        """
        nodal_ix, edge_ix = self.topo.expand_facets(facets)
        facet_ix = facets

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

    def __init__(self, topo, element):

        self.topo = topo
        self.element = element

        self._nodal_dofs = np.reshape(
            np.arange(element.nodal_dofs * topo.nvertices, dtype=np.int64),
            (element.nodal_dofs, topo.nvertices),
            order='F')
        offset = element.nodal_dofs * topo.nvertices

        # edge dofs
        if element.dim == 3:
            self._edge_dofs = np.reshape(
                np.arange(element.edge_dofs * topo.nedges,
                          dtype=np.int64),
                (element.edge_dofs, topo.nedges),
                order='F') + offset
            offset += element.edge_dofs * topo.nedges
        else:
            self._edge_dofs = np.empty((0, 0), dtype=np.int64)

        # facet dofs
        self._facet_dofs = np.reshape(
            np.arange(element.facet_dofs * topo.nfacets,
                      dtype=np.int64),
            (element.facet_dofs, topo.nfacets),
            order='F') + offset
        offset += element.facet_dofs * topo.nfacets

        # interior dofs
        self._interior_dofs = np.reshape(
            np.arange(element.interior_dofs * topo.nelements, dtype=np.int64),
            (element.interior_dofs, topo.nelements),
            order='F') + offset

        # global numbering
        self._element_dofs = np.zeros((0, topo.nelements), dtype=np.int64)

        # nodal dofs
        for itr in range(topo.t.shape[0]):
            self._element_dofs = np.vstack((
                self._element_dofs,
                self._nodal_dofs[:, topo.t[itr]]
            ))

        # edge dofs
        if element.dim == 3:
            for itr in range(topo.t2e.shape[0]):
                self._element_dofs = np.vstack((
                    self._element_dofs,
                    self._edge_dofs[:, topo.t2e[itr]]
                ))

        # facet dofs
        if element.dim >= 2:
            for itr in range(topo.t2f.shape[0]):
                self._element_dofs = np.vstack((
                    self._element_dofs,
                    self._facet_dofs[:, topo.t2f[itr]]
                ))

        # interior dofs
        self._element_dofs = np.vstack((self._element_dofs,
                                       self._interior_dofs))

        # total dofs
        self.N = np.max(self._element_dofs) + 1

    def _dofnames_to_rows(self, dofnames, skip=False):

        if isinstance(dofnames, str):
            dofnames = [dofnames]

        if skip is True:
            check = lambda x, y: x not in y
        else:
            check = lambda x, y: x in y

        n_nodal = self._nodal_dofs.shape[0]
        n_facet = self._facet_dofs.shape[0]
        n_edge = self._edge_dofs.shape[0]
        n_interior = self._interior_dofs.shape[0]

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
