import warnings
from typing import NamedTuple, Dict, Union, List
from copy import deepcopy

import numpy as np
from numpy import ndarray

from skfem.element import Element
from skfem.mesh import Mesh


class DofsView:

    nodal_ix: ndarray = None
    edge_ix: ndarray = None
    facet_ix: ndarray = None
    interior_ix: ndarray = None

    def __init__(self,
                 obj,
                 nodal_ix=None,
                 edge_ix=None,
                 facet_ix=None,
                 interior_ix=None):
        self.nodal_ix = np.empty((0,), dtype=np.int64)\
            if nodal_ix is None else nodal_ix
        self.edge_ix = np.empty((0,), dtype=np.int64)\
            if edge_ix is None else edge_ix
        self.facet_ix = np.empty((0,), dtype=np.int64)\
            if facet_ix is None else facet_ix
        self.interior_ix = np.empty((0,), dtype=np.int64)\
            if interior_ix is None else interior_ix
        self._obj = obj

    def all(self, keep_dofnames=None):
        if keep_dofnames is not None:
            i1, i2, i3, i4 = self._dofnames_to_ix(keep_dofnames)
        else:
            i1, i2, i3, i4 = 4 * (slice(None),)
        return np.unique(
            np.concatenate(
                (
                    self._obj._nodal_dofs[i1][:, self.nodal_ix].flatten(),
                    self._obj._facet_dofs[i2][:, self.facet_ix].flatten(),
                    self._obj._edge_dofs[i3][:, self.edge_ix].flatten(),
                    self._obj._interior_dofs[i4][:, self.interior_ix].flatten()
                )
            )
        )

    @property
    def nodal(self):
        return self._by_name(self._nodal_dofs,
                             ix=self.nodal_ix)

    @property
    def facet(self):
        return self._by_name(self._facet_dofs,
                             off=self._nodal_dofs.shape[0],
                             ix=self.facet_ix)

    @property
    def edge(self):
        return self._by_name(self._edge_dofs,
                             off=(self._nodal_dofs.shape[0]
                                  + self._facet_dofs.shape[0]),
                             ix=self.edge_ix)

    @property
    def interior(self):
        return self._by_name(self._interior_dofs,
                             off=(self._nodal_dofs.shape[0]
                                  + self._facet_dofs.shape[0]
                                  + self._edge_dofs.shape[0]),
                             ix=self.interior_ix)

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __or__(self, other):
        """For merging two sets of DOF's."""
        return DofsView(
            self._obj,
            np.union1d(self.nodal_ix, other.nodal_ix),
            np.union1d(self.edge_ix, other.edge_ix),
            np.union1d(self.facet_ix, other.facet_ix),
            np.union1d(self.interior_ix, other.interior_ix)
        )


class Dofs:
    """An object containing a set of degree-of-freedom indices."""

    _nodal_dofs: ndarray = None
    _edge_dofs: ndarray = None
    _facet_dofs: ndarray = None
    _interior_dofs: ndarray = None

    _element_dofs: ndarray = None
    N: int = 0

    topo: Mesh = None
    element: Element = None

    def _by_name(self, dofs, off=0, ix=None):

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

    @property
    def nodal(self):
        return self._by_name(self._nodal_dofs)

    @property
    def facet(self):
        return self._by_name(self._facet_dofs,
                             off=self._nodal_dofs.shape[0])

    @property
    def edge(self):
        return self._by_name(self._edge_dofs,
                             off=(self._nodal_dofs.shape[0]
                                  + self._facet_dofs.shape[0]))

    @property
    def interior(self):
        return self._by_name(self._interior_dofs,
                             off=(self._nodal_dofs.shape[0]
                                  + self._facet_dofs.shape[0]
                                  + self._edge_dofs.shape[0]))

    def get_facet_dofs(self, facets):

        nodal_ix, edge_ix = self.topo.expand_facets(facets)
        facet_ix = facets

        return DofsView(
            self,
            nodal_ix=nodal_ix,
            edge_ix=edge_ix,
            facet_ix=facet_ix
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

    def _dofnames_to_ix(self, dofnames):

        if isinstance(dofnames, str):
            dofnames = [dofnames]

        n_nodal = self._nodal_dofs.shape[0]
        n_facet = self._facet_dofs.shape[0]
        n_edge = self._edge_dofs.shape[0]
        n_interior = self._interior_dofs.shape[0]

        i1 = []
        for i in range(n_nodal):
            if self.element.dofnames[i] in dofnames:
                i1.append(i)

        i2 = []
        for i in range(n_facet):
            if self.element.dofnames[i + n_nodal] in dofnames:
                i2.append(i)

        i3 = []
        for i in range(n_edge):
            if self.element.dofnames[i + n_nodal + n_facet] in dofnames:
                i3.append(i)

        i4 = []
        for i in range(n_interior):
            if self.element.dofnames[i + n_nodal + n_facet + n_edge] in dofnames:
                i4.append(i)

        return (
            i1 if len(i1) > 0 else slice(0, 0),
            i2 if len(i2) > 0 else slice(0, 0),
            i3 if len(i3) > 0 else slice(0, 0),
            i4 if len(i4) > 0 else slice(0, 0)
        )

    def all(self, keep_dofnames=None):
        if keep_dofnames is not None:
            i1, i2, i3, i4 = self._dofnames_to_ix(keep_dofnames)
        else:
            i1, i2, i3, i4 = 4 * (slice(None),)
        return np.unique(
            np.concatenate(
                (
                    self._nodal_dofs[i1].flatten(),
                    self._facet_dofs[i2].flatten(),
                    self._edge_dofs[i3].flatten(),
                    self._interior_dofs[i4].flatten()
                )
            )
        )

    def __array__(self):
        return self.all()
