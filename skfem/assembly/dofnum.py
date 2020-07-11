import numpy as np

from numpy import ndarray


class Dofnum:
    """Numbering for the global degrees-of-freedom."""

    nodal_dofs: ndarray = None
    edge_dofs: ndarray = None
    facet_dofs: ndarray = None
    interior_dofs: ndarray = None

    element_dofs: ndarray = None
    N: int = 0

    def __init__(self, topo, element):

        self.topo = topo
        self.element = element

        self.nodal_dofs = np.reshape(
            np.arange(element.nodal_dofs * topo.nvertices, dtype=np.int64),
            (element.nodal_dofs, topo.nvertices),
            order='F')
        offset = element.nodal_dofs * topo.nvertices

        # edge dofs
        if element.dim == 3:
            self.edge_dofs = np.reshape(
                np.arange(element.edge_dofs * topo.nedges,
                          dtype=np.int64),
                (element.edge_dofs, topo.nedges),
                order='F') + offset
            offset += element.edge_dofs * topo.nedges
        else:
            self.edge_dofs = np.empty((0, 0))

        # facet dofs
        self.facet_dofs = np.reshape(
            np.arange(element.facet_dofs * topo.nfacets,
                      dtype=np.int64),
            (element.facet_dofs, topo.nfacets),
            order='F') + offset
        offset += element.facet_dofs * topo.nfacets

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
        if element.dim == 3:
            for itr in range(topo.t2e.shape[0]):
                self.element_dofs = np.vstack((
                    self.element_dofs,
                    self.edge_dofs[:, topo.t2e[itr]]
                ))

        # facet dofs
        if element.dim >= 2:
            for itr in range(topo.t2f.shape[0]):
                self.element_dofs = np.vstack((
                    self.element_dofs,
                    self.facet_dofs[:, topo.t2f[itr]]
                ))

        # interior dofs
        self.element_dofs = np.vstack((self.element_dofs, self.interior_dofs))

        # total dofs
        self.N = np.max(self.element_dofs) + 1
