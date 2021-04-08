from dataclasses import dataclass, replace
from typing import Tuple, Type, Union, Optional, Dict, Callable, List
from collections import namedtuple
from itertools import dropwhile
from warnings import warn

import numpy as np
from numpy import ndarray
from scipy.spatial import cKDTree

from ..element import (Element, ElementHex1, ElementQuad1, ElementQuad2,
                       ElementTetP1, ElementTriP1, ElementTriP2, ElementLineP1,
                       ElementTetP2, ElementHex2, BOUNDARY_ELEMENT_MAP)


@dataclass(repr=False)
class Mesh:

    doflocs: ndarray
    t: ndarray
    _boundaries: Optional[Dict[str, ndarray]] = None
    _subdomains: Optional[Dict[str, ndarray]] = None
    elem: Type[Element] = Element
    affine: bool = False
    validate: bool = False  # unused; for backwards compatibility
    # Some parts of the library, most notably the normal vector construction in
    # ElementGlobal._eval_dofs, assume that the element indices are ascending
    # because this leads to consistent normal vectors for both elements sharing
    # a facet.  Therefore, the element indices are sorted in a triangle mesh.
    # However, some algorithms (e.g., adaptive refinement) require switching
    # off this behaviour and, hence, this flag exists.
    sort_t: bool = False

    @property
    def p(self):
        return self.doflocs

    @property
    def dofs(self):
        from skfem.assembly import Dofs
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
        return self._subdomains

    @property
    def boundaries(self):
        return self._boundaries

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

    def define_boundary(self, name: str,
                        test: Callable[[ndarray], ndarray],
                        boundaries_only: bool = True):
        """Define a named boundary via function handle.

        Parameters
        ----------
        name
            Name of the boundary.
        test
            A function which returns True for facet midpoints belonging to the
            boundary.
        boundaries_only
            If True, include only facets on the boundary of the mesh.

        """
        warn("Mesh.define_boundary is deprecated and will be removed in the "
             "next major release.", DeprecationWarning)
        if self._boundaries is None:
            self._boundaries = {}
        self._boundaries[name] = self.facets_satisfying(test, boundaries_only)

    def with_boundaries(self,
                        boundaries: Dict[str, Callable[[ndarray], ndarray]]):
        """Return a copy of the mesh with named boundaries.

        Parameters
        ----------
        boundaries
            A dictionary of lambda functions with the names of the boundaries
            as keys.  The midpoint of the facet should return ``True`` for the
            corresponding lambda function if the facet belongs to the boundary.

        """
        return replace(
            self,
            _boundaries={
                **({} if self._boundaries is None else self._boundaries),
                **{name: self.facets_satisfying(test, True)
                   for name, test in boundaries.items()}
            },
        )

    def with_subdomains(self,
                        subdomains: Dict[str, Callable[[ndarray], ndarray]]):
        """Return a copy of the mesh with named subdomains.

        Parameters
        ----------
        boundaries
            A dictionary of lambda functions with the names of the subdomains
            as keys.  The midpoint of the element should return ``True`` for
            the corresponding lambda function if the element belongs to the
            subdomain.

        """
        return replace(
            self,
            _subdomains={
                **({} if self._subdomains is None else self._subdomains),
                **{name: self.elements_satisfying(test)
                   for name, test in subdomains.items()},
            }
        )

    def boundary_nodes(self) -> ndarray:
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:, self.boundary_facets()])

    def interior_nodes(self) -> ndarray:
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0, self.p.shape[1]),
                            self.boundary_nodes())

    def nodes_satisfying(self,
                         test: Callable[[ndarray], ndarray],
                         boundaries_only: bool = False) -> ndarray:
        """Return nodes that satisfy some condition.

        Parameters
        ----------
        test
            A function which returns ``True`` for the set of nodes that are to
            be included in the return set.
        boundaries_only
            If ``True``, include only boundary facets.

        """
        nodes = np.nonzero(test(self.p))[0]
        if boundaries_only:
            nodes = np.intersect1d(nodes, self.boundary_nodes())
        return nodes

    def facets_satisfying(self,
                          test: Callable[[ndarray], ndarray],
                          boundaries_only: bool = False) -> ndarray:
        """Return facets whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            A function which returns ``True`` for the facet midpoints that are
            to be included in the return set.
        boundaries_only
            If ``True``, include only boundary facets.

        """
        midp = [np.sum(self.p[itr, self.facets], axis=0) / self.facets.shape[0]
                for itr in range(self.dim())]
        facets = np.nonzero(test(np.array(midp)))[0]
        if boundaries_only:
            facets = np.intersect1d(facets, self.boundary_facets())
        return facets

    def elements_satisfying(self,
                            test: Callable[[ndarray], ndarray]) -> ndarray:
        """Return elements whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            A function which returns ``True`` for the element midpoints that
            are to be included in the return set.

        """
        midp = [np.sum(self.p[itr, self.t], axis=0) / self.t.shape[0]
                for itr in range(self.dim())]
        return np.nonzero(test(np.array(midp)))[0]

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
        from skfem.mapping import MappingAffine, MappingIsoparametric
        if not hasattr(self, '_cached_mapping'):
            if self.affine:
                self._cached_mapping = MappingAffine(self)
            else:
                # TODO make MappingIsoparametric compatible with self
                FakeMesh = namedtuple(
                    'FakeMesh',
                    ['p', 't', 'facets', 't2f', 'f2t', 'dim']
                )
                fakemesh = FakeMesh(
                    self.doflocs,
                    self.dofs.element_dofs,
                    self.facets,
                    self.t2f,
                    self.f2t,
                    lambda: self.dim(),
                )
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
            self.elem.refdom.facets,
        )

    def _init_edges(self):
        """Initialize ``self.edges``."""
        self._edges, self._t2e = self.build_entities(
            self.t,
            self.elem.refdom.edges,
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
        if self.sort_t:
            self.t = np.sort(self.t, axis=0)

        if not isinstance(self.doflocs, ndarray):
            # for backwards compatibility: support standard lists
            self.doflocs = np.array(self.doflocs, dtype=np.float64)

        if not isinstance(self.t, ndarray):
            # for backwards compatibility: support standard lists
            self.t = np.array(self.t, dtype=np.int64)

        M = self.elem.refdom.nnodes

        if self.nnodes > M:
            # reorder DOFs to the expected format: vertex DOFs are first
            p, t = self.doflocs, self.t
            t_nodes = t[:M]
            uniq, ix = np.unique(t_nodes, return_inverse=True)
            self.t = (np.arange(len(uniq), dtype=np.int64)[ix]
                      .reshape(t_nodes.shape))
            doflocs = np.hstack((
                p[:, uniq],
                np.zeros((p.shape[0], np.max(t) + 1 - len(uniq))),
            ))
            doflocs[:, self.dofs.element_dofs[M:].flatten('F')] =\
                p[:, t[M:].flatten('F')]
            self.doflocs = doflocs

        # C_CONTIGUOUS is more performant in dimension-based slices
        if self.doflocs.flags['F_CONTIGUOUS']:
            if self.doflocs.shape[1] > 1000:
                warn("Transforming over 1000 vertices to C_CONTIGUOUS.")
            self.doflocs = np.ascontiguousarray(self.doflocs)

        if self.t.flags['F_CONTIGUOUS']:
            if self.t.shape[1] > 1000:
                warn("Transforming over 1000 elements to C_CONTIGUOUS.")
            self.t = np.ascontiguousarray(self.t)

    def __add__(self, other):
        """Join two meshes."""
        if not isinstance(other, type(self)):
            raise TypeError("Can only join meshes with same type.")
        p = np.hstack((self.p, other.p))
        t = np.hstack((self.t, other.t + self.p.shape[1]))
        tmp = np.ascontiguousarray(p.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        p = p[:, ixa]
        t = ixb[t]
        cls = type(self)
        return cls(p, t)

    def __repr__(self):
        return "{} mesh with {} vertices and {} elements.".format(
            self.elem.refdom.name,
            self.nvertices,
            self.nelements,
        )

    def __str__(self):
        return self.__repr__()

    def save(self,
             filename: str,
             point_data: Optional[Dict[str, ndarray]] = None,
             **kwargs) -> None:
        """Export the mesh and fields using meshio.

        Parameters
        ----------
        filename
            The output filename, with suffix determining format;
            e.g. .msh, .vtk, .xdmf
        point_data
            Data related to the vertices of the mesh.

        """
        from skfem.io.meshio import to_file
        return to_file(self, filename, point_data, **kwargs)

    @classmethod
    def load(cls, filename):
        from skfem.io.meshio import from_file
        return from_file(filename)

    @classmethod
    def from_dict(cls, data):
        """For backwards compatibility."""
        if 'p' not in data or 't' not in data:
            raise ValueError("Dictionary must contain keys 'p' and 't'.")
        else:
            data['p'] = np.ascontiguousarray(np.array(data['p']).T)
            data['t'] = np.ascontiguousarray(np.array(data['t']).T)
        if 'boundaries' in data and data['boundaries'] is not None:
            data['boundaries'] = {k: np.array(v)
                                  for k, v in data['boundaries'].items()}
        if 'subdomains' in data and data['subdomains'] is not None:
            data['subdomains'] = {k: np.array(v)
                                  for k, v in data['subdomains'].items()}
        data['doflocs'] = data.pop('p')
        data['_subdomains'] = data.pop('subdomains')
        data['_boundaries'] = data.pop('boundaries')
        return cls(**data)

    def to_dict(self) -> Dict[str, Optional[Dict[str, List[float]]]]:
        """For backwards compatibility."""
        boundaries: Optional[Dict[str, List[float]]] = None
        subdomains: Optional[Dict[str, List[float]]] = None
        if self.boundaries is not None:
            boundaries = {k: v.tolist() for k, v in self.boundaries.items()}
        if self.subdomains is not None:
            subdomains = {k: v.tolist() for k, v in self.subdomains.items()}
        return {
            'p': self.p.T.tolist(),
            't': self.t.T.tolist(),
            'boundaries': boundaries,
            'subdomains': subdomains,
        }

    @classmethod
    def from_mesh(cls, mesh):
        """Reuse an existing mesh by adding nodes.

        Parameters
        ----------
        mesh
            The mesh used in the initialization.  Connectivity of the new mesh
            will match ``mesh.t``.

        """
        from skfem.assembly import Dofs

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
        m = self
        if isinstance(times_or_ix, int):
            for _ in range(times_or_ix):
                m = m._uniform()
        else:
            m = m._adaptive(times_or_ix)
        return m

    def scaled(self, factors):
        """Return a new mesh with scaled dimensions.

        Parameters
        ----------
        factors
            Scale each dimension by a factor.

        """
        if isinstance(factors, float):
            # for backwards compatibility
            factors = self.doflocs.shape[0] * [factors]
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

    def mirrored(self,
                 normal: Tuple[float, ...],
                 point: Optional[Tuple[float, ...]] = None):
        """Return a mesh mirrored with respect to a normal.

        Meant to be combined with the other methods to build more general
        meshes, e.g.,

        >>> from skfem import MeshTet
        >>> m1 = MeshTet()
        >>> m2 = m1.mirrored((1, 0, 0))
        >>> m3 = m1.mirrored((0, 1, 0))
        >>> m4 = m1.mirrored((0, 0, 1))
        >>> m = m1 + m2 + m3 + m4
        >>> (m.nvertices, m.nelements)
        (20, 20)

        Parameters
        ----------
        normal
            The normal vector of the mirror plane.
        point
            An optional point through which the plane passes. By default, the
            point corresponds to the origin.

        """
        if point is None:
            point = (0,) * self.dim()

        p = self.p.copy()
        p0 = np.array(point)
        n = np.array(normal)
        n = n / np.linalg.norm(n)
        p += - 2. * np.dot(n, p - p0[:, None]) * n[:, None] + p0[:, None]

        return replace(
            self,
            doflocs=p,
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
    def build_entities(t, indices, sort=True):
        """Build low dimensional topological entities."""
        indexing = np.hstack(tuple([t[ix] for ix in indices]))
        sorted_indexing = np.sort(indexing, axis=0)

        sorted_indexing, ixa, ixb = np.unique(sorted_indexing,
                                              axis=1,
                                              return_index=True,
                                              return_inverse=True)
        mapping = ixb.reshape((len(indices), t.shape[1]))

        if sort:
            return np.ascontiguousarray(sorted_indexing), mapping

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

    def param(self) -> float:
        """Return mesh parameter, viz the length of the longest edge."""
        raise NotImplementedError

    def _reix(self, ix: ndarray) -> Tuple[ndarray, ndarray]:
        """Connect ``self.p`` based on the indices ``ix``."""
        ixuniq = np.unique(ix)
        t = np.zeros(np.max(ix) + 1, dtype=np.int64)
        t[ixuniq] = np.arange(len(ixuniq), dtype=np.int64)
        return self.p[:, ixuniq], t[ix]

    def remove_elements(self, element_indices: ndarray):
        """Construct a new mesh by removing elements.

        Parameters
        ----------
        element_indices
            List of element indices to remove.

        """
        p, t = self._reix(np.delete(self.t, element_indices, axis=1))
        return replace(
            self,
            doflocs=p,
            t=t,
        )

    def element_finder(self, mapping=None):
        """Return a function handle from location to element index.

        Parameters
        ----------
        mapping
            The affine mapping for the mesh.

        """
        raise NotImplementedError


@dataclass(repr=False)
class Mesh2D(Mesh):

    def param(self) -> float:
        return np.max(
            np.linalg.norm(np.diff(self.p[:, self.facets], axis=1), axis=0)
        )

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        """For meshio which appends :math:`z = 0` to 2D meshes."""
        return p[:, :2]

    def _repr_svg_(self) -> str:
        from skfem.visuals.svg import draw
        return draw(self, nrefs=2, boundaries_only=True)


@dataclass(repr=False)
class Mesh3D(Mesh):

    def param(self) -> float:
        return np.max(
            np.linalg.norm(np.diff(self.p[:, self.edges], axis=1), axis=0)
        )

    def edges_satisfying(self, test: Callable[[ndarray], bool]) -> ndarray:
        """Return edges whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            Evaluates to 1 or ``True`` for edge midpoints of the edges
            belonging to the output set.

        """
        return np.nonzero(test(self.p[:, self.edges].mean(1)))[0]

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

    def interior_edges(self) -> ndarray:
        """Return an array of interior edge indices."""
        return np.setdiff1d(np.arange(self.edges.shape[1], dtype=np.int64),
                            self.boundary_edges())


@dataclass(repr=False)
class MeshTri1(Mesh2D):

    doflocs: ndarray = np.array([[0., 0.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 1.]], dtype=np.float64).T
    t: ndarray = np.array([[0, 1, 2],
                           [1, 3, 2]], dtype=np.int64).T
    elem: Type[Element] = ElementTriP1
    affine: bool = True
    sort_t: bool = True

    @classmethod
    def init_tensor(cls: Type, x: ndarray, y: ndarray):
        r"""Initialize a tensor product mesh.

        The mesh topology is as follows::

            *---------------*
            |'-.|'-.|`'---._|
            |---+---+-------|
            |\  |\  |'.     |
            | \ | \ |  '-.  |
            |  \|  \|     '.|
            *---------------*

        Parameters
        ----------
        x
            The nodal coordinates in dimension `x`.
        y
            The nodal coordinates in dimension `y`.

        """
        npx = len(x)
        npy = len(y)
        X, Y = np.meshgrid(np.sort(x), np.sort(y))
        p = np.vstack((X.flatten('F'), Y.flatten('F')))
        ix = np.arange(npx * npy)
        nt = (npx - 1) * (npy - 1)
        t = np.zeros((3, 2 * nt))
        ix = ix.reshape(npy, npx, order='F').copy()
        t[0, :nt] = (ix[0:(npy-1), 0:(npx-1)].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())
        t[1, :nt] = (ix[1:npy, 0:(npx-1)].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())
        t[2, :nt] = (ix[1:npy, 1:npx].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())
        t[0, nt:] = (ix[0:(npy-1), 0:(npx-1)].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())
        t[1, nt:] = (ix[0:(npy-1), 1:npx].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())
        t[2, nt:] = (ix[1:npy, 1:npx].reshape(nt, 1, order='F')
                     .copy()
                     .flatten())

        return cls(p, t.astype(np.int64))

    @classmethod
    def init_symmetric(cls: Type) -> Mesh2D:
        r"""Initialize a symmetric mesh of the unit square.

        The mesh topology is as follows::

            *------------*
            |\          /|
            |  \      /  |
            |    \  /    |
            |     *      |
            |    /  \    |
            |  /      \  |
            |/          \|
            O------------*

        """
        p = np.array([[0., 1., 1., 0., .5],
                      [0., 0., 1., 1., .5]], dtype=np.float64)
        t = np.array([[0, 1, 4],
                      [1, 2, 4],
                      [2, 3, 4],
                      [0, 3, 4]], dtype=np.int64).T
        return cls(p, t)

    @classmethod
    def init_sqsymmetric(cls: Type) -> Mesh2D:
        r"""Initialize a symmetric mesh of the unit square.

        The mesh topology is as follows::

            *------*------*
            |\     |     /|
            |  \   |   /  |
            |    \ | /    |
            *------*------*
            |    / | \    |
            |  /   |   \  |
            |/     |     \|
            O------*------*

        """
        p = np.array([[0., .5, 1., 0., .5, 1., 0., .5, 1.],
                      [0., 0., 0., .5, .5, .5, 1., 1., 1.]], dtype=np.float64)
        t = np.array([[0, 1, 4],
                      [1, 2, 4],
                      [2, 4, 5],
                      [0, 3, 4],
                      [3, 4, 6],
                      [4, 6, 7],
                      [4, 7, 8],
                      [4, 5, 8]], dtype=np.int64).T
        return cls(p, t)

    @classmethod
    def init_lshaped(cls: Type) -> Mesh2D:
        r"""Initialize a mesh for the L-shaped domain.

        The mesh topology is as follows::

            *-------*
            | \     |
            |   \   |
            |     \ |
            |-------O-------*
            |     / | \     |
            |   /   |   \   |
            | /     |     \ |
            *---------------*

        """
        p = np.array([[0., 1., 0., -1.,  0., -1., -1.,  1.],
                      [0., 0., 1.,  0., -1., -1.,  1., -1.]], dtype=np.float64)
        t = np.array([[0, 1, 7],
                      [0, 2, 6],
                      [0, 6, 3],
                      [0, 7, 4],
                      [0, 4, 5],
                      [0, 3, 5]], dtype=np.int64).T
        return cls(p, t)

    @classmethod
    def init_circle(cls: Type,
                    nrefs: int = 3) -> Mesh2D:
        r"""Initialize a circle mesh.

        Works by repeatedly refining the following mesh and moving
        new nodes to the boundary::

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
        nrefs
            Number of refinements, by default 3.

        """
        p = np.array([[0., 0.],
                      [1., 0.],
                      [0., 1.],
                      [-1., 0.],
                      [0., -1.]], dtype=np.float64).T
        t = np.array([[0, 1, 2],
                      [0, 1, 4],
                      [0, 2, 3],
                      [0, 3, 4]], dtype=np.int64).T
        m = cls(p, t)
        for _ in range(nrefs):
            m = m.refined()
            D = m.boundary_nodes()
            tmp = m.p
            tmp[:, D] = tmp[:, D] / np.linalg.norm(tmp[:, D], axis=0)
            m = replace(m, doflocs=tmp)
        return m

    def _uniform(self):

        p = self.doflocs
        t = self.t
        sz = p.shape[1]
        t2f = self.t2f.copy() + sz
        return replace(
            self,
            doflocs=np.hstack((p, p[:, self.facets].mean(axis=1))),
            t=np.hstack((
                np.vstack((t[0], t2f[0], t2f[2])),
                np.vstack((t[1], t2f[0], t2f[1])),
                np.vstack((t[2], t2f[2], t2f[1])),
                np.vstack((t2f[0], t2f[1], t2f[2])),
            )),
            _boundaries=None,
            _subdomains=None,
        )

    @staticmethod
    def _adaptive_sort_mesh(p, t):
        """Make (0, 2) the longest edge in t."""
        l01 = np.sqrt(np.sum((p[:, t[0]] - p[:, t[1]]) ** 2, axis=0))
        l12 = np.sqrt(np.sum((p[:, t[1]] - p[:, t[2]]) ** 2, axis=0))
        l02 = np.sqrt(np.sum((p[:, t[0]] - p[:, t[2]]) ** 2, axis=0))

        ix01 = (l01 > l02) * (l01 > l12)
        ix12 = (l12 > l01) * (l12 > l02)

        # row swaps
        tmp = t[2, ix01]
        t[2, ix01] = t[1, ix01]
        t[1, ix01] = tmp

        tmp = t[0, ix12]
        t[0, ix12] = t[1, ix12]
        t[1, ix12] = tmp

        return t

    @staticmethod
    def _adaptive_find_facets(m, marked_elems):
        """Find the facets to split."""
        facets = np.zeros(m.facets.shape[1], dtype=np.int64)
        facets[m.t2f[:, marked_elems].flatten('F')] = 1
        prev_nnz = -1e10

        while np.count_nonzero(facets) - prev_nnz > 0:
            prev_nnz = np.count_nonzero(facets)
            t2facets = facets[m.t2f]
            t2facets[2, t2facets[0] + t2facets[1] > 0] = 1
            facets[m.t2f[t2facets == 1]] = 1

        return facets

    @staticmethod
    def _adaptive_split_elements(m, facets):
        """Define new elements."""
        ix = (-1) * np.ones(m.facets.shape[1], dtype=np.int64)
        ix[facets == 1] = (np.arange(np.count_nonzero(facets))
                           + m.p.shape[1])
        ix = ix[m.t2f]

        red = (ix[0] >= 0) * (ix[1] >= 0) * (ix[2] >= 0)
        blue1 = (ix[0] == -1) * (ix[1] >= 0) * (ix[2] >= 0)
        blue2 = (ix[0] >= 0) * (ix[1] == -1) * (ix[2] >= 0)
        green = (ix[0] == -1) * (ix[1] == -1) * (ix[2] >= 0)
        rest = (ix[0] == -1) * (ix[1] == -1) * (ix[2] == -1)

        # new red elements
        t_red = np.hstack((
            np.vstack((m.t[0, red], ix[0, red], ix[2, red])),
            np.vstack((m.t[1, red], ix[0, red], ix[1, red])),
            np.vstack((m.t[2, red], ix[1, red], ix[2, red])),
            np.vstack((ix[1, red], ix[2, red], ix[0, red])),
        ))

        # new blue elements
        t_blue1 = np.hstack((
            np.vstack((m.t[1, blue1], m.t[0, blue1], ix[2, blue1])),
            np.vstack((m.t[1, blue1], ix[1, blue1], ix[2, blue1])),
            np.vstack((m.t[2, blue1], ix[2, blue1], ix[1, blue1])),
        ))

        t_blue2 = np.hstack((
            np.vstack((m.t[0, blue2], ix[0, blue2], ix[2, blue2])),
            np.vstack((ix[2, blue2], ix[0, blue2], m.t[1, blue2])),
            np.vstack((m.t[2, blue2], ix[2, blue2], m.t[1, blue2])),
        ))

        # new green elements
        t_green = np.hstack((
            np.vstack((m.t[1, green], ix[2, green], m.t[0, green])),
            np.vstack((m.t[2, green], ix[2, green], m.t[1, green])),
        ))

        # new nodes
        p = .5 * (m.p[:, m.facets[0, facets == 1]] +
                  m.p[:, m.facets[1, facets == 1]])

        return (
            np.hstack((m.p, p)),
            np.hstack((m.t[:, rest], t_red, t_blue1, t_blue2, t_green)),
        )

    def _adaptive(self, marked):

        sorted_mesh = replace(
            self,
            t=self._adaptive_sort_mesh(self.p, self.t),
            sort_t=False,
        )
        facets = self._adaptive_find_facets(sorted_mesh, marked)
        doflocs, t = self._adaptive_split_elements(sorted_mesh, facets)

        return replace(
            self,
            doflocs=doflocs,
            t=t,
            _boundaries=None,
            _subdomains=None,
        )

    def element_finder(self, mapping=None):

        if mapping is None:
            mapping = self._mapping()

        tree = cKDTree(np.mean(self.p[:, self.t], axis=1).T)

        def finder(x, y):
            ix = tree.query(np.array([x, y]).T, 5)[1].flatten()
            X = mapping.invF(np.array([x, y])[:, None], ix)
            inside = (
                (X[0] >= 0) *
                (X[1] >= 0) *
                (1 - X[0] - X[1] >= 0)
            )
            return np.array([ix[np.argmax(inside, axis=0)]]).flatten()

        return finder


@dataclass(repr=False)
class MeshQuad1(Mesh2D):

    doflocs: ndarray = np.array([[0., 0.],
                                 [1., 0.],
                                 [1., 1.],
                                 [0., 1.]], dtype=np.float64).T
    t: ndarray = np.array([[0, 1, 2, 3]], dtype=np.int64).T
    elem: Type[Element] = ElementQuad1

    def _uniform(self):

        p = self.doflocs
        t = self.t
        sz = p.shape[1]
        t2f = self.t2f.copy() + sz
        mid = np.arange(t.shape[1], dtype=np.int64) + np.max(t2f) + 1
        return replace(
            self,
            doflocs=np.hstack((
                p,
                p[:, self.facets].mean(axis=1),
                p[:, self.t].mean(axis=1),
            )),
            t=np.hstack((
                np.vstack((t[0], t2f[0], mid, t2f[3])),
                np.vstack((t2f[0], t[1], t2f[1], mid)),
                np.vstack((mid, t2f[1], t[2], t2f[2])),
                np.vstack((t2f[3], mid, t2f[2], t[3])),
            )),
            _boundaries=None,
            _subdomains=None,
        )

    @classmethod
    def init_tensor(cls: Type,
                    x: ndarray,
                    y: ndarray):
        """Initialize a tensor product mesh.

        The mesh topology is as follows::

            *-------------*
            |   |  |      |
            |---+--+------|
            |   |  |      |
            |   |  |      |
            |   |  |      |
            *-------------*

        Parameters
        ----------
        x
            The nodal coordinates in dimension `x`.
        y
            The nodal coordinates in dimension `y`.

        """
        npx = len(x)
        npy = len(y)
        X, Y = np.meshgrid(np.sort(x), np.sort(y))
        p = np.vstack((X.flatten('F'), Y.flatten('F')))
        ix = np.arange(npx * npy)
        nt = (npx - 1) * (npy - 1)
        t = np.zeros((4, nt))
        ix = ix.reshape(npy, npx, order='F').copy()
        t[0] = (ix[0:(npy-1), 0:(npx-1)].reshape(nt, 1, order='F')
                .copy()
                .flatten())
        t[1] = (ix[1:npy, 0:(npx-1)].reshape(nt, 1, order='F')
                .copy()
                .flatten())
        t[2] = (ix[1:npy, 1:npx].reshape(nt, 1, order='F')
                .copy()
                .flatten())
        t[3] = (ix[0:(npy-1), 1:npx].reshape(nt, 1, order='F')
                .copy()
                .flatten())
        return cls(p, t.astype(np.int64))

    def to_meshtri(self, x: Optional[ndarray] = None):
        """Split each quadrilateral into two triangles."""
        t = np.hstack((self.t[[0, 1, 3]], self.t[[1, 2, 3]]))

        subdomains = None
        if self.subdomains:
            subdomains = {k: np.concatenate((v, v + self.t.shape[1]))
                          for k, v in self.subdomains.items()}

        mesh = MeshTri1(self.doflocs, t)

        boundaries = None
        if self.boundaries:
            boundaries = {}
            for k in self.boundaries:
                slots = enumerate(mesh.facets.T)
                boundaries[k] = np.array([
                    next(dropwhile(lambda slot: not(np.array_equal(f,
                                                                   slot[1])),
                                   slots))[0]
                    for f in self.facets.T[np.sort(self.boundaries[k])]])

        if self._subdomains or self._boundaries:
            mesh = replace(
                mesh,
                _boundaries=boundaries,
                _subdomains=subdomains,
            )

        if x is not None:
            if len(x) == self.t.shape[1]:
                # preserve elemental constant functions
                X = np.concatenate((x, x))
            else:
                raise Exception("The parameter x must have one value per "
                                "element.")
            return mesh, X
        return mesh

    def element_finder(self, mapping=None):
        """Transform to :class:`skfem.MeshTri` and return its finder."""
        tri_finder = self.to_meshtri().element_finder()

        def finder(*args):
            return tri_finder(*args) % self.t.shape[1]

        return finder


@dataclass(repr=False)
class MeshTri2(MeshTri1):

    elem: Type[Element] = ElementTriP2
    affine: bool = False
    sort_t: bool = False

    @classmethod
    def init_circle(cls: Type,
                    nrefs: int = 3) -> Mesh2D:
        m = MeshTri1.init_circle(nrefs=nrefs)
        M = cls.from_mesh(m)
        D = M.dofs.get_facet_dofs(M.boundary_facets()).flatten()
        doflocs = M.doflocs.copy()
        doflocs[:, D] /= np.linalg.norm(doflocs[:, D], axis=0)
        return replace(M, doflocs=doflocs)


@dataclass(repr=False)
class MeshQuad2(MeshQuad1):

    elem: Type[Element] = ElementQuad2


@dataclass(repr=False)
class MeshLine1(Mesh):

    doflocs: ndarray = np.array([[0., 1.]], dtype=np.float64)
    t: ndarray = np.array([[0], [1]], dtype=np.int64)
    elem: Type[Element] = ElementLineP1
    affine: bool = True

    def __post_init__(self):

        if len(self.doflocs.shape) == 1:
            # support flat arrays
            self.doflocs = np.array([self.doflocs])

        if self.t.shape[1] != self.doflocs.shape[1] - 1:
            # fill self.t assuming ascending self.doflocs if not provided
            tmp = np.arange(self.doflocs.shape[1] - 1, dtype=np.int64)
            self.t = np.vstack((tmp, tmp + 1))

        super().__post_init__()

    def __mul__(self, other):
        return MeshQuad1.init_tensor(self.p[0], other.p[0])

    def _uniform(self):
        p, t = self.doflocs, self.t

        newp = np.hstack((p, p[:, t].mean(axis=1)))
        newt = np.empty((t.shape[0], 2 * t.shape[1]),
                        dtype=t.dtype)
        newt[0, ::2] = t[0]
        newt[0, 1::2] = p.shape[1] + np.arange(t.shape[1])
        newt[1, ::2] = newt[0, 1::2]
        newt[1, 1::2] = t[1]

        return replace(
            self,
            doflocs=newp,
            t=newt,
            _boundaries=None,
            _subdomains=None,
        )

    def _adaptive(self, marked):
        p, t = self.doflocs, self.t

        mid = range(len(marked)) + np.max(t) + 1
        nonmarked = np.setdiff1d(np.arange(t.shape[1]), marked)
        newp = np.hstack((p, p[:, t[:, marked]].mean(1)))
        newt = np.vstack((t[0, marked], mid))
        newt = np.hstack((t[:, nonmarked],
                          newt,
                          np.vstack((mid, t[1, marked]))))

        return replace(
            self,
            doflocs=newp,
            t=newt,
        )

    def param(self):
        return np.max(np.abs(self.p[0, self.t[1]] - self.p[0, self.t[0]]))

    def element_finder(self, mapping=None):
        ix = np.argsort(self.p)

        def finder(x):
            maxix = (x == np.max(self.p))
            x[maxix] = x[maxix] - 1e-10  # special case in np.digitize
            return np.argmax(np.digitize(x, self.p[0, ix[0]])[:, None]
                             == self.t[0], axis=1)

        return finder

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p[:, :1]


@dataclass(repr=False)
class MeshTet1(Mesh3D):

    doflocs: ndarray = np.array([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 1., 1.],
                                 [1., 0., 1.],
                                 [1., 1., 0.],
                                 [1., 1., 1.]], dtype=np.float64).T
    t: ndarray = np.array([[0, 1, 2, 3],
                           [3, 5, 1, 7],
                           [2, 3, 6, 7],
                           [2, 3, 1, 7],
                           [1, 2, 4, 7]], dtype=np.int64).T
    elem: Type[Element] = ElementTetP1
    affine: bool = True

    def element_finder(self, mapping=None):

        if mapping is None:
            mapping = self._mapping()

        tree = cKDTree(np.mean(self.p[:, self.t], axis=1).T)

        def finder(x, y, z):
            ix = tree.query(np.array([x, y, z]).T, 5)[1].flatten()
            X = mapping.invF(np.array([x, y, z])[:, None], ix)
            inside = (
                (X[0] >= 0) *
                (X[1] >= 0) *
                (X[2] >= 0) *
                (1 - X[0] - X[1] - X[2] >= 0)
            )
            return np.array([ix[np.argmax(inside, axis=0)]]).flatten()

        return finder

    def _uniform(self):
        t = self.t
        p = self.p
        sz = p.shape[1]
        t2e = self.t2e.copy() + sz

        # new vertices are the midpoints of edges
        newp = np.hstack((p, p[:, self.edges].mean(axis=1)))

        # compute middle pyramid diagonal lengths and choose shortest
        d1 = ((newp[0, t2e[2]] - newp[0, t2e[4]]) ** 2 +
              (newp[1, t2e[2]] - newp[1, t2e[4]]) ** 2)
        d2 = ((newp[0, t2e[1]] - newp[0, t2e[3]]) ** 2 +
              (newp[1, t2e[1]] - newp[1, t2e[3]]) ** 2)
        d3 = ((newp[0, t2e[0]] - newp[0, t2e[5]]) ** 2 +
              (newp[1, t2e[0]] - newp[1, t2e[5]]) ** 2)
        I1 = d1 < d2
        I2 = d1 < d3
        I3 = d2 < d3
        c1 = I1 * I2
        c2 = (~I1) * I3
        c3 = (~I2) * (~I3)

        # splitting the pyramid in the middle;
        # diagonals are [2,4], [1,3] and [0,5]
        newt = np.hstack((
            np.vstack((t[0], t2e[0], t2e[2], t2e[3])),
            np.vstack((t[1], t2e[0], t2e[1], t2e[4])),
            np.vstack((t[2], t2e[1], t2e[2], t2e[5])),
            np.vstack((t[3], t2e[3], t2e[4], t2e[5])),
            np.vstack((t2e[2, c1], t2e[4, c1], t2e[0, c1], t2e[1, c1])),
            np.vstack((t2e[2, c1], t2e[4, c1], t2e[0, c1], t2e[3, c1])),
            np.vstack((t2e[2, c1], t2e[4, c1], t2e[1, c1], t2e[5, c1])),
            np.vstack((t2e[2, c1], t2e[4, c1], t2e[3, c1], t2e[5, c1])),
            np.vstack((t2e[1, c2], t2e[3, c2], t2e[0, c2], t2e[4, c2])),
            np.vstack((t2e[1, c2], t2e[3, c2], t2e[4, c2], t2e[5, c2])),
            np.vstack((t2e[1, c2], t2e[3, c2], t2e[5, c2], t2e[2, c2])),
            np.vstack((t2e[1, c2], t2e[3, c2], t2e[2, c2], t2e[0, c2])),
            np.vstack((t2e[0, c3], t2e[5, c3], t2e[1, c3], t2e[4, c3])),
            np.vstack((t2e[0, c3], t2e[5, c3], t2e[4, c3], t2e[3, c3])),
            np.vstack((t2e[0, c3], t2e[5, c3], t2e[3, c3], t2e[2, c3])),
            np.vstack((t2e[0, c3], t2e[5, c3], t2e[2, c3], t2e[1, c3])),
        ))

        return replace(
            self,
            doflocs=newp,
            t=newt,
            _boundaries=None,
            _subdomains=None,
        )

    @classmethod
    def init_tensor(cls: Type,
                    x: ndarray,
                    y: ndarray,
                    z: ndarray):
        """Initialize a tensor product mesh.

        Parameters
        ----------
        x
            The nodal coordinates in dimension `x`.
        y
            The nodal coordinates in dimension `y`.
        z
            The nodal coordinates in dimension `z`.

        """
        npx = len(x)
        npy = len(y)
        npz = len(z)
        X, Y, Z = np.meshgrid(np.sort(x), np.sort(y), np.sort(z))
        p = np.vstack((X.flatten('F'), Y.flatten('F'), Z.flatten('F')))
        ix = np.arange(npx * npy * npz)
        ne = (npx - 1) * (npy - 1) * (npz - 1)
        t = np.zeros((8, ne))
        ix = ix.reshape(npy, npx, npz, order='F').copy()
        t[0] = (ix[0:(npy - 1), 0:(npx - 1), 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[1] = (ix[1:npy, 0:(npx - 1), 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[2] = (ix[0:(npy - 1), 1:npx, 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[3] = (ix[0:(npy - 1), 0:(npx - 1), 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[4] = (ix[1:npy, 1:npx, 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[5] = (ix[1:npy, 0:(npx - 1), 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[6] = (ix[0:(npy - 1), 1:npx, 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[7] = (ix[1:npy, 1:npx, 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())

        T = np.zeros((4, 0))
        T = np.hstack((T, t[[0, 1, 5, 7]]))
        T = np.hstack((T, t[[0, 1, 4, 7]]))
        T = np.hstack((T, t[[0, 2, 4, 7]]))
        T = np.hstack((T, t[[0, 3, 5, 7]]))
        T = np.hstack((T, t[[0, 2, 6, 7]]))
        T = np.hstack((T, t[[0, 3, 6, 7]]))

        return cls(p, T.astype(np.int64))

    @classmethod
    def init_ball(cls: Type,
                  nrefs: int = 3):
        """Initialize a ball mesh.

        Parameters
        ----------
        nrefs
            Number of refinements, by default 3.

        """
        p = np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.],
                      [-1., 0., 0.],
                      [0., -1., 0.],
                      [0., 0., -1.]], dtype=np.float64).T
        t = np.array([[0, 1, 2, 3],
                      [0, 4, 5, 6],
                      [0, 1, 2, 6],
                      [0, 1, 3, 5],
                      [0, 2, 3, 4],
                      [0, 4, 5, 3],
                      [0, 4, 6, 2],
                      [0, 5, 6, 1]], dtype=np.int64).T
        m = cls(p, t)
        for _ in range(nrefs):
            m = m.refined()
            D = m.boundary_nodes()
            tmp = m.p
            tmp[:, D] = tmp[:, D] / np.linalg.norm(tmp[:, D], axis=0)
            m = replace(m, doflocs=tmp)
        return m


@dataclass(repr=False)
class MeshHex1(Mesh3D):
    """Hexahedral mesh.

    If `t` is provided, order of vertices in each element should match the
    numbering::

            2---6
           /   /|
          4---7 3
          |   |/
          1---5

    """

    doflocs: ndarray = np.array([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 1., 1.],
                                 [1., 0., 1.],
                                 [1., 1., 0.],
                                 [1., 1., 1.]], dtype=np.float64).T
    t: ndarray = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64).T
    elem: Type[Element] = ElementHex1

    def _init_facets(self):
        """Initialize ``self.facets`` without sorting"""
        self._facets, self._t2f = self.build_entities(
            self.t,
            self.elem.refdom.facets,
            sort=False,
        )

    def _uniform(self):
        p = self.doflocs
        t = self.t
        sz = p.shape[1]
        t2e = self.t2e.copy() + sz
        t2f = self.t2f.copy() + np.max(t2e) + 1
        mid = np.arange(self.t.shape[1], dtype=np.int64) + np.max(t2f) + 1

        doflocs = np.hstack((
            p,
            .5 * np.sum(p[:, self.edges], axis=1),
            .25 * np.sum(p[:, self.facets], axis=1),
            .125 * np.sum(p[:, t], axis=1),
        ))
        t = np.hstack((
            np.vstack((t[0], t2e[0], t2e[1], t2e[2],
                       t2f[0], t2f[2], t2f[1], mid)),
            np.vstack((t2e[0], t[1], t2f[0], t2f[2],
                       t2e[3], t2e[4], mid, t2f[4])),
            np.vstack((t2e[1], t2f[0], t[2], t2f[1],
                       t2e[5], mid, t2e[6], t2f[3])),
            np.vstack((t2e[2], t2f[2], t2f[1], t[3],
                       mid, t2e[7], t2e[8], t2f[5])),
            np.vstack((t2f[0], t2e[3], t2e[5], mid,
                       t[4], t2f[4], t2f[3], t2e[9])),
            np.vstack((t2f[2], t2e[4], mid, t2e[7],
                       t2f[4], t[5], t2f[5], t2e[10])),
            np.vstack((t2f[1], mid, t2e[6], t2e[8],
                       t2f[3], t2f[5], t[6], t2e[11])),
            np.vstack((mid, t2f[4], t2f[3], t2f[5],
                       t2e[9], t2e[10], t2e[11], t[7]))
        ))
        return replace(
            self,
            doflocs=doflocs,
            t=t,
            _boundaries=None,
            _subdomains=None,
        )

    @classmethod
    def init_tensor(cls: Type,
                    x: ndarray,
                    y: ndarray,
                    z: ndarray):
        """Initialize a tensor product mesh.

        Parameters
        ----------
        x
            The nodal coordinates in dimension `x`.
        y
            The nodal coordinates in dimension `y`.
        z
            The nodal coordinates in dimension `z`.

        """
        npx = len(x)
        npy = len(y)
        npz = len(z)
        X, Y, Z = np.meshgrid(np.sort(x), np.sort(y), np.sort(z))
        p = np.vstack((X.flatten('F'), Y.flatten('F'), Z.flatten('F')))
        ix = np.arange(npx * npy * npz)
        ne = (npx - 1) * (npy - 1) * (npz - 1)
        t = np.zeros((8, ne))
        ix = ix.reshape(npy, npx, npz, order='F').copy()
        t[0] = (ix[0:(npy - 1), 0:(npx - 1), 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[1] = (ix[1:npy, 0:(npx - 1), 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[2] = (ix[0:(npy - 1), 1:npx, 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[3] = (ix[0:(npy - 1), 0:(npx - 1), 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[4] = (ix[1:npy, 1:npx, 0:(npz - 1)]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[5] = (ix[1:npy, 0:(npx - 1), 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[6] = (ix[0:(npy - 1), 1:npx, 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        t[7] = (ix[1:npy, 1:npx, 1:npz]
                .reshape(ne, 1, order='F')
                .copy()
                .flatten())
        return cls(p, t.astype(np.int64))

    def to_meshtet(self):
        """Split each hexahedron into six tetrahedra."""
        t = np.hstack((
            self.t[[0, 1, 3, 4]],
            self.t[[0, 3, 2, 4]],
            self.t[[2, 3, 4, 6]],
            self.t[[3, 4, 6, 7]],
            self.t[[3, 4, 5, 7]],
            self.t[[1, 3, 4, 5]],
        ))

        return MeshTet1(self.doflocs, t)

    def element_finder(self, mapping=None):
        """Transform to :class:`skfem.MeshTet` and return its finder."""
        tet_finder = self.to_meshtet().element_finder()

        def finder(*args):
            return tet_finder(*args) % self.t.shape[1]

        return finder


@dataclass(repr=False)
class MeshTet2(MeshTet1):

    elem: Type[Element] = ElementTetP2

    @classmethod
    def init_ball(cls: Type, nrefs: int = 3) -> Mesh3D:
        m = MeshTet1.init_ball(nrefs=nrefs)
        M = cls.from_mesh(m)
        D = M.dofs.get_facet_dofs(M.boundary_facets()).flatten()
        doflocs = M.doflocs.copy()
        doflocs[:, D] /= np.linalg.norm(doflocs[:, D], axis=0)
        return replace(M, doflocs=doflocs)


@dataclass(repr=False)
class MeshHex2(MeshHex1):

    elem: Type[Element] = ElementHex2
