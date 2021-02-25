import warnings
from dataclasses import dataclass, replace
from typing import NamedTuple, Tuple, Type

import numpy as np
from numpy import ndarray

from ..assembly import Dofs
from ..element import (Element, ElementHex1, ElementQuad1, ElementQuad2,
                       ElementTetP1, ElementTriP1, ElementTriP2)


@dataclass
class Geometry:

    doflocs: ndarray
    t: ndarray
    elem: Type[Element] = Element
    affine: bool = False

    @property
    def p(self):
        return self.doflocs

    @property
    def dofs(self):
        if not hasattr(self, '_dofs'):
            self._dofs = Dofs(self, self.elem())
        return self._dofs

    @property
    def refdom(self):
        return self.elem.refdom

    @property
    def brefdom(self):
        return self.elem.refdom.brefdom

    def _mapping(self):
        # TODO cache mapping
        from skfem.mapping import MappingAffine, MappingIsoparametric

        class FakeMesh(NamedTuple):
            p: ndarray
            t: ndarray
            facets: ndarray

        fakemesh = FakeMesh(
            self.doflocs,
            self.dofs.element_dofs,
            self.facets,
        )

        if self.affine:
            return MappingAffine(fakemesh)

        return MappingIsoparametric(
            fakemesh,
            self.elem(),
            self.bndelem,
        )

    @property
    def bndelem(self):

        from skfem.element import (ElementHex1, ElementHex2, ElementLineP1,
                                   ElementLineP2, ElementQuad1, ElementQuad2,
                                   ElementTetP1, ElementTetP2, ElementTriP1,
                                   ElementTriP2)

        BOUNDARY_ELEMENT_MAP = {  # TODO move to Element attributes
            ElementTriP1: ElementLineP1,
            ElementTriP2: ElementLineP2,
            ElementQuad1: ElementLineP1,
            ElementQuad2: ElementLineP2,
            ElementTetP1: ElementTriP1,
            ElementTetP2: ElementTriP2,
            ElementHex1: ElementQuad1,
            ElementHex2: ElementQuad2,
        }

        return BOUNDARY_ELEMENT_MAP[self.elem]()

    def dim(self):
        return self.elem.refdom.dim()

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

    @property
    def nnodes(self):
        return self.t.shape[0]

    def _init_facets(self):
        self._facets, self._t2f = self.build_entities(
            self.t,
            self.elem.refdom.facets
        )

    def _init_edges(self):
        self._edges, self._t2e = self.build_entities(
            self.t,
            self.elem.refdom.edges
        )

    @property
    def subdomains(self):
        return None  # TODO

    @property
    def boundaries(self):
        return None

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
            self._f2t = self.build_inverse(self.t, self.t2f)
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

        indexing = np.hstack(tuple([t[ix] for ix in indices]))
        sorted_indexing = np.sort(indexing, axis=0)

        sorted_indexing, ixa, ixb = np.unique(sorted_indexing,
                                              axis=1,
                                              return_index=True,
                                              return_inverse=True)
        mapping = ixb.reshape((len(indices), t.shape[1]))

        return np.ascontiguousarray(indexing[:, ixa]), mapping

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

    def boundary_facets(self) -> ndarray:
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1] == -1)[0]

    def boundary_edges(self) -> ndarray:
        """Return an array of boundary edge indices."""
        facets = self.boundary_facets()
        boundary_edges = np.sort(np.hstack(
            tuple([np.vstack((self.facets[itr, facets],
                              self.facets[(itr + 1) % self.facets.shape[0],
                              facets]))
                   for itr in range(self.facets.shape[0])])).T, axis=1)
        edge_candidates = np.unique(self.t2e[:, self.f2t[0, facets]])
        A = self.edges[:, edge_candidates].T
        B = boundary_edges
        dims = A.max(0) + 1
        ix = np.where(np.in1d(np.ravel_multi_index(A.T, dims),
                              np.ravel_multi_index(B.T, dims)))[0]
        return edge_candidates[ix]

    def _expand_facets(self, ix: ndarray) -> Tuple[ndarray, ndarray]:

        vertices = np.unique(self.facets[:, ix].flatten())

        if self.dim() == 3:
            edge_candidates = self.t2e[:, self.f2t[0, ix]].flatten()
            # subset of edges that share all points with the given facets
            subset = np.nonzero(
                np.prod(np.isin(self.edges[:, edge_candidates],
                                self.facets[:, ix].flatten()),
                        axis=0)
            )[0]
            edges = np.intersect1d(self.boundary_edges(),
                                   edge_candidates[subset])
        else:
            edges = np.array([], dtype=np.int64)

        return vertices, edges

    def __post_init__(self):

        M = self.elem.refdom.nnodes
        if self.nnodes > M:  # TODO check that works for 3D quadratic
            # reorder DOFs to the expected format: vertex DOFs are first
            p, t = self.doflocs, self.t
            _t = t[:M]
            uniq, ix = np.unique(_t, return_inverse=True)
            rest = np.setdiff1d(np.arange(np.max(t) + 1, dtype=np.int64),
                                uniq)
            self.t = np.arange(len(uniq), dtype=np.int64)[ix].reshape(_t.shape)
            _p = np.hstack((
                p[:, uniq],
                np.zeros((p.shape[0], np.max(t) + 1 - len(uniq))),
            ))
            _p[:, self.dofs.element_dofs[M:].flatten('F')] =\
                p[:, t[M:].flatten('F')]
            self.doflocs = _p

    @classmethod
    def load(cls, filename):
        from skfem.io.meshio import from_file
        return from_file(filename)

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        """Fallback for 3D meshes."""
        return p

    def refined(self, times: int = 1):
        m = self
        for _ in range(times):
            m = m._uniform()
        return m

    @classmethod
    def from_mesh(cls, mesh):
        mapping = mesh._mapping()
        nelem = cls.elem
        dofs = Dofs(mesh, nelem())
        locs = mapping.F(nelem.doflocs.T)
        doflocs = np.zeros((locs.shape[0], dofs.N))

        # match mapped dofs and global dof numbering
        for itr in range(locs.shape[0]):
            for jtr in range(dofs.element_dofs.shape[0]):
                doflocs[itr, dofs.element_dofs[jtr]] = locs[itr, :, jtr]

        return cls(
            doflocs=doflocs,
            t=mesh.t,
        )

    @classmethod
    def init_refdom(cls):
        return cls(cls.elem.refdom.p, cls.elem.refdom.t)

    def _splitref(self, nrefs: int = 1):

        cls = type(self)
        m = cls.init_refdom().refined(nrefs)
        X = m.p
        x = self._mapping().F(m.p)

        # create connectivity for the new mesh
        nt = self.nelements
        t = np.tile(m.t, (1, nt))
        dt = np.max(t)
        t += ((dt + 1)
              * (np.tile(np.arange(nt), (m.t.shape[0] * m.t.shape[1], 1))
                 .flatten('F')
                 .reshape((-1, m.t.shape[0])).T))

        if X.shape[0] == 1:
            p = np.array([x.flatten()])
        else:
            p = x[0].flatten()
            for itr in range(len(x) - 1):
                p = np.vstack((p, x[itr + 1].flatten()))

        return cls(p, t)


@dataclass
class Geometry2D(Geometry):

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        """For meshio which appends :math:`z = 0` to 2D meshes."""
        return p[:, :2]

    def _repr_svg_(self) -> str:
        from skfem.visuals.svg import draw
        return draw(self, nrefs=2, boundaries_only=True)


@dataclass
class MeshTri1(Geometry2D):

    elem: Type[Element] = ElementTriP1
    affine: bool = True

    def _uniform(self):
        p = self.doflocs
        t = self.t
        t2f = self.t2f
        sz = p.shape[1]
        return replace(
            self,
            doflocs=np.hstack((p, p[:, self.facets].mean(axis=1))),
            t=np.hstack((
                np.vstack((t[0], t2f[0] + sz, t2f[2] + sz)),
                np.vstack((t[1], t2f[0] + sz, t2f[1] + sz)),
                np.vstack((t[2], t2f[2] + sz, t2f[1] + sz)),
                np.vstack((t2f[0] + sz, t2f[1] + sz, t2f[2] + sz)),
            )),
        )


@dataclass
class MeshQuad1(Geometry2D):

    elem: Type[Element] = ElementQuad1

    def _uniform(self):
        p = self.doflocs
        t = self.t
        t2f = self.t2f
        sz = p.shape[1]
        mid = np.arange(t.shape[1], dtype=np.int64) + np.max(t2f) + sz + 1
        return replace(
            self,
            doflocs=np.hstack((
                p,
                p[:, self.facets].mean(axis=1),
                p[:, self.t].mean(axis=1),
            )),
            t=np.hstack((
                np.vstack((t[0], t2f[0] + sz, mid, t2f[3] + sz)),
                np.vstack((t2f[0] + sz, t[1], t2f[1] + sz, mid)),
                np.vstack((mid, t2f[1] + sz, t[2], t2f[2] + sz)),
                np.vstack((t2f[3] + sz, mid, t2f[2] + sz, t[3])),
            )),
        )


@dataclass
class MeshTri2(MeshTri1):

    elem: Type[Element] = ElementTriP2
    affine: bool = False


@dataclass
class MeshQuad2(MeshQuad1):

    elem: Type[Element] = ElementQuad2


@dataclass
class MeshTet1(Geometry):

    elem: Type[Element] = ElementTetP1
    affine: bool = True


@dataclass
class MeshHex1(Geometry):

    elem: Type[Element] = ElementHex1
