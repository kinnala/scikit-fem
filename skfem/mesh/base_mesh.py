from dataclasses import dataclass, replace
from typing import Tuple, Type, Union
from collections import namedtuple

import numpy as np
from numpy import ndarray

from ..assembly import Dofs
from ..element import (Element, ElementHex1, ElementQuad1, ElementQuad2,
                       ElementTetP1, ElementTriP1, ElementTriP2,
                       BOUNDARY_ELEMENT_MAP)
from ..mapping import MappingAffine, MappingIsoparametric


@dataclass
class BaseMesh:

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

    @property
    def bndelem(self):
        return BOUNDARY_ELEMENT_MAP[self.elem]()

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

    @property
    def subdomains(self):
        return None

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

    def dim(self):
        return self.elem.refdom.dim()

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
        """Return vertices and edges corresponding to given facet indices.

        Parameters
        ----------
        ix
            An array of facet indices.

        """
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

    def _mapping(self):
        """Return a default reference mapping for the mesh."""
        if not hasattr(self, '_cached_mapping'):
            fakemesh = namedtuple('Mesh', ['p', 't', 'facets'])(
                self.doflocs,
                self.dofs.element_dofs,
                self.facets,
            )
            if self.affine:
                self._cached_mapping = MappingAffine(fakemesh)
            else:
                self._cached_mapping = MappingIsoparametric(
                    fakemesh,
                    self.elem(),
                    self.bndelem,
                )
        return self._cached_mapping

    def _init_facets(self):
        """Initialize ``self.facets``."""
        self._facets, self._t2f = self.build_entities(
            self.t,
            self.elem.refdom.facets
        )

    def _init_edges(self):
        """Initialize ``self.edges``."""
        self._edges, self._t2e = self.build_entities(
            self.t,
            self.elem.refdom.edges
        )

    def __post_init__(self):
        """Support node orders used in external formats.

        We expect ``self.doflocs`` to be ordered based on the
        degrees-of-freedom in :class:`skfem.assembly.Dofs`.  External formats
        for high order meshes commonly use a less strict ordering scheme and
        the extra nodes are described as additional rows in ``self.t``.  This
        method attempts to accommodate external formas by reordering
        ``self.doflocs`` and changing the indices in ``self.t``.

        """
        M = self.elem.refdom.nnodes
        if self.nnodes > M:  # TODO check that works for 3D quadratic
            # reorder DOFs to the expected format: vertex DOFs are first
            p, t = self.doflocs, self.t
            _t = t[:M]
            uniq, ix = np.unique(_t, return_inverse=True)
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

    @classmethod
    def from_mesh(cls, mesh):
        """Reuse an existing mesh by adding nodes.

        Parameters
        ----------
        mesh
            The mesh used in the initialization.  Connectivity of the new mesh
            will match ``mesh.t``.

        """
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
        """Initialize a mesh corresponding to the reference domain."""
        return cls(cls.elem.refdom.p, cls.elem.refdom.t)

    def refined(self, times_or_ix: Union[int, ndarray] = 1):
        """Return a refined mesh.

        Parameters
        ----------
        times_or_ix
            Either an integer giving the number of uniform refinements or an
            array of element indices for adaptive refinement.

        """
        if isinstance(times_or_ix, int):
            m = self
            for _ in range(times_or_ix):
                m = m._uniform()
        elif isinstance(times_or_ix, ndarray):
            m = m._adaptive(times_or_ix)
        else:
            raise NotImplementedError
        return m

    def scaled(self, factors):
        """Return a new mesh with scaled dimensions.

        Parameters
        ----------
        factors
            Scale each dimension by a factor.

        """
        return replace(
            self,
            doflocs=np.array([self.doflocs[itr] * factors[itr]
                              for itr in range(len(factors))]),
        )

    def translated(self, diffs):
        """Return a new translated mesh.

        Parameters
        ----------
        diffs
            Translate the mesh by a vector. Must have same size as the mesh
            dimension.

        """
        return replace(
            self,
            doflocs=np.array([self.doflocs[itr] + diffs[itr]
                              for itr in range(len(diffs))]),
        )

    def _uniform(self):
        """Perform a single uniform refinement."""
        raise NotImplementedError

    def _adaptive(self, ix: ndarray):
        """Adaptively refine the given set of elements."""
        raise NotImplementedError

    def _splitref(self, nrefs: int = 1):
        """Split mesh into separate nonconnected elements and refine.

        Used for visualization purposes.

        Parameters
        ----------
        nrefs
            The number of refinements.

        """
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

    @staticmethod
    def build_entities(t, indices):
        """Build low dimensional topological entities."""
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
        """Build inverse mapping from low dimensional topological entities."""
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

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        """Fallback for 3D meshes."""
        return p


@dataclass
class BaseMesh2D(BaseMesh):

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        """For meshio which appends :math:`z = 0` to 2D meshes."""
        return p[:, :2]

    def _repr_svg_(self) -> str:
        from skfem.visuals.svg import draw
        return draw(self, nrefs=2, boundaries_only=True)


@dataclass
class MeshTri1(BaseMesh2D):

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
class MeshQuad1(BaseMesh2D):

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
class MeshTet1(BaseMesh):

    elem: Type[Element] = ElementTetP1
    affine: bool = True


@dataclass
class MeshHex1(BaseMesh):

    elem: Type[Element] = ElementHex1
