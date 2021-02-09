from typing import Optional, Type, Tuple

import numpy as np

from numpy import ndarray

from dataclasses import dataclass, replace


@dataclass
class Graph:

    t: ndarray

    @property
    def nelements(self):
        return self.t.shape[1]

    @property
    def nvertices(self):
        return np.max(self.t) + 1

    @property
    def nfacets(self):
        return self.facets.shape[1]

    @property
    def nedges(self):
        return self.edges.shape[1]

    def _init_facets(self):
        self._facets, self._t2f = Graph.build_entities(
            self.t,
            self._facet_indices
        )

    def _init_edges(self):
        self._edges, self._t2e = Graph.build_entities(
            self.t,
            self._edge_indices
        )

    @property
    def facets(self):
        if not hasattr(self, '_facets'):
            self._init_facets()
        return self._facets

    @property
    def t2f(self):
        if not hasattr(self, '_t2f'):
            self._init_facets()
        return self._t2f

    @property
    def f2t(self):
        if not hasattr(self, '_f2t'):
            self._f2t = Graph.build_inverse(self.t, self.t2f)
        return self._f2t

    @property
    def edges(self):
        if not hasattr(self, '_edges'):
            self._init_edges()
        return self._edges

    @property
    def t2e(self):
        if not hasattr(self, '_t2e'):
            self._init_edges()
        return self._t2e

    @staticmethod
    def build_entities(t, indices):

        indexing = np.sort(np.hstack(
            tuple([t[entity] for entity in indices])
        ), axis=0)

        indexing, ixa, ixb = np.unique(indexing,
                                       axis=1,
                                       return_index=True,
                                       return_inverse=True)
        mapping = ixb.reshape((len(indices), t.shape[1]))

        return np.ascontiguousarray(indexing), mapping

    @staticmethod
    def build_inverse(t, mapping):

        e = mapping.flatten(order='C')
        tix = np.tile(np.arange(t.shape[1]), (1, t.shape[0]))[0]

        e_first, ix_first = np.unique(e, return_index=True)
        e_last, ix_last = np.unique(e[::-1], return_index=True)
        ix_last = e.shape[0] - ix_last - 1

        inverse = np.zeros((2, np.max(mapping) + 1), dtype=np.int64)
        inverse[0, e_first] = tix[ix_first]
        inverse[1, e_last] = tix[ix_last]
        inverse[1, np.nonzero(inverse[0] == inverse[1])[0]] = -1

        return inverse

    @property
    def _edge_indices(self):

        if self.t.shape[0] == 4:
            return [
                [0, 1],
                [1, 2],
                [0, 2],
                [0, 3],
                [1, 3],
                [2, 3],
            ]
        elif self.t.shape[0] == 8:
            return [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 4],
                [1, 5],
                [2, 4],
                [2, 6],
                [3, 5],
                [3, 6],
                [4, 7],
                [5, 7],
                [6, 7],
            ]
        else:
            raise Exception("Cannot build edges for the given mesh.")

    @property
    def _facet_indices(self):

        if self.t.shape[0] == 3:
            return [
                [0, 1],
                [1, 2],
                [0, 2],
            ]
        elif self.t.shape[0] == 4:
            return [
                [0, 1],
                [1, 2],
                [2, 3],
                [0, 3],
            ]
        elif self.t.shape[0] == 6:
            return [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ]
        elif self.t.shape[0] == 8:
            return [
                [0, 1, 4, 2],
                [0, 2, 6, 3],
                [0, 3, 5, 1],
                [2, 4, 7, 6],
                [1, 5, 7, 4],
                [3, 6, 7, 5],
            ]
        else:
            raise Exception("Cannot build facets for the given mesh.")

    def boundary_facets(self) -> ndarray:
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1] == -1)[0]

    def _expand_facets(self, facets: ndarray) -> Tuple[ndarray, ndarray]:
        """Find vertices and edges corresponding to given facets."""
        vertices = np.unique(self.facets[:, facets].flatten())
        edges = np.array([], dtype=np.int64)
        return vertices, edges


@dataclass
class Grid(Graph):

    p: ndarray
    elem: 'Element'
    element_dofs: Optional[ndarray] = None
    doflocs: Optional[ndarray] = None

    @property
    def dofs(self):

        from skfem.assembly import Dofs

        if not hasattr(self, '_dofs'):
            self._dofs = Dofs(self, self.elem)
        return self._dofs

    @property
    def refdom(self):  # todo
        return self.elem.mesh_type.refdom

    @property
    def brefdom(self):  # todo
        return self.elem.mesh_type.brefdom

    def _mapping(self):

        from skfem.mapping import MappingIsoparametric
        from skfem.element import (ElementTriP1, ElementTriP2,
                                   ElementLineP1, ElementLineP2,
                                   ElementQuad1, ElementQuad2,
                                   ElementTetP1, ElementTetP2,
                                   ElementHex1, ElementHex2)

        BOUNDARY_ELEMENT_MAP = {
            ElementTriP1: ElementLineP1,
            ElementTriP2: ElementLineP2,
            ElementQuad1: ElementLineP1,
            ElementQuad2: ElementLineP2,
            ElementTetP1: ElementTriP1,
            ElementTetP2: ElementTriP2,
            ElementHex1: ElementQuad1,
            ElementHex2: ElementQuad2,
        }

        return MappingIsoparametric(
            replace(self,
                    t=self.element_dofs,
                    p=self.doflocs) if self.doflocs is not None else self,
            self.elem,
            BOUNDARY_ELEMENT_MAP[type(self.elem)]()
        )

    def dim(self):
        return self.elem.dim


class BaseMesh:

    grid: Grid

    def __init__(self, t, p, elem, element_dofs=None, doflocs=None):

        self.grid = Grid(
            t=t,
            p=p,
            elem=elem,
            element_dofs=element_dofs,
            doflocs=doflocs,
        )

    @classmethod
    def load(cls, filename):

        from skfem.io.meshio import from_file

        return from_file(filename)

    def __getattr__(self, item):
        return getattr(self.grid, item)


class BaseMesh2D(BaseMesh):

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p[:, :2]


class MeshTri1(BaseMesh2D):

    def __init__(self, p, t, **kwargs):

        from skfem.element import ElementTriP1

        super(MeshTri1, self).__init__(
            t,
            p,
            ElementTriP1(),
        )

    @classmethod
    def init_circle(cls, Nrefs: int = 3):
        r"""Initialize a circle mesh.

        Repeatedly refines the following mesh and moves new nodes to the
        boundary::

                   *
                 / | \
               /   |   \
             /     |     \
            *------O------*
             \     |     /
               \   |   /
                 \ | /
                   *

        Parameters
        ----------
        Nrefs
            Number of refinements, by default 3.

        """
        p = np.array([[0., 0.],
                      [1., 0.],
                      [0., 1.],
                      [-1., 0.],
                      [0., -1.]]).T
        t = np.array([[0, 1, 2],
                      [0, 1, 4],
                      [0, 2, 3],
                      [0, 3, 4]], dtype=np.int64).T
        m = cls(p, t)
        for _ in range(Nrefs):
            m = m.refined()
            D = m.boundary_nodes()
            m.p[:, D] = m.p[:, D] / np.linalg.norm(m.p[:, D], axis=0)
        return m


class MeshTri2(BaseMesh2D):

    def __init__(self, doflocs, element_dofs, **kwargs):

        from skfem.element import ElementTriP2

        dofs, ix = np.unique(element_dofs[:3], return_inverse=True)
        p = doflocs[:, dofs]
        t = (np.arange(len(dofs), dtype=np.int64)[ix]
             .reshape(element_dofs[:3].shape))
        super(MeshTri2, self).__init__(
            t,
            p,
            ElementTriP2(),
            element_dofs,
            doflocs,
        )

    @classmethod
    def init_circle(cls, Nrefs=3):
        m = MeshTri1.init_circle(Nrefs)
        m = cls(m.p, m.t)
        D = m._basis.get_dofs(m.boundary_facets()).flatten()
        m._mesh.p[:, D] =\
            m._mesh.p[:, D] / np.linalg.norm(m._mesh.p[:, D], axis=0)
        return m
