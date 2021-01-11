import warnings
from typing import Dict, Optional, Tuple, Type, TypeVar, Union, Callable

import numpy as np
from numpy import ndarray

MeshType = TypeVar('MeshType', bound='Mesh')
DimTuple = Union[Tuple[float],
                 Tuple[float, float],
                 Tuple[float, float, float]]


class Mesh:
    """A finite element mesh.

    Typically initialized via one of the following subclasses:

    - :class:`~skfem.mesh.MeshTri`, triangular mesh
    - :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh
    - :class:`~skfem.mesh.MeshTet`, tetrahedral mesh
    - :class:`~skfem.mesh.MeshHex`, hexahedral mesh
    - :class:`~skfem.mesh.MeshLine`, one-dimensional mesh

    Attributes
    ----------
    p
        The vertices of the mesh (dim x Nvertices). Each column corresponds to
        a point.
    t
        The element connectivity (dim x Nelements). Each column corresponds to
        a element and contains column indices to `self.p`.
    subdomains
        Named subsets of elements. Empty if not loaded from an external format.
    boundaries
        Named subsets of boundary facets. Empty if not loaded from an external
        format or defined via :meth:`~skfem.mesh.Mesh.define_boundary`.

    """

    refdom: str = "none"
    brefdom: str = "none"
    meshio_type: str = "none"
    name: str = "Abstract"

    p = np.array([], dtype=np.float64)
    t = np.array([], dtype=np.int64)
    facets: ndarray
    t2f: ndarray
    f2t: ndarray
    edges: ndarray
    t2e: ndarray

    subdomains: Optional[Dict[str, ndarray]] = None
    boundaries: Optional[Dict[str, ndarray]] = None

    def __init__(self, *args, **kwargs):
        """Check that p and t are C_CONTIGUOUS as this leads
        to better performance."""
        if self.p is not None:
            if not isinstance(self.p, ndarray):
                self.p = np.array(self.p, dtype=np.float64)
            if self.p.flags['F_CONTIGUOUS']:
                if self.p.shape[1] > 1000:
                    warnings.warn("Mesh.__init__(): Transforming "
                                  "over 100 vertices to C_CONTIGUOUS.")
                self.p = np.ascontiguousarray(self.p)
        if self.t is not None:
            if not isinstance(self.t, ndarray):
                self.t = np.array(self.t, dtype=np.int64)
            if self.t.flags['F_CONTIGUOUS']:
                if self.t.shape[1] > 1000:
                    warnings.warn("Mesh.__init__(): Transforming "
                                  "over 100 elements to C_CONTIGUOUS.")
                self.t = np.ascontiguousarray(self.t)
        # transform lists to ndarrays
        if self.boundaries is not None:
            for k, v in self.boundaries.items():
                if not isinstance(v, ndarray):
                    self.boundaries[k] = np.array(v, dtype=np.int64)
        if self.subdomains is not None:
            for k, v in self.subdomains.items():
                if not isinstance(v, ndarray):
                    self.subdomains[k] = np.array(v, dtype=np.int64)

    @property
    def nelements(self):
        return int(self.t.shape[1])

    @property
    def nvertices(self):
        return int(np.max(self.t) + 1)

    @property
    def nfacets(self):
        return int(self.facets.shape[1])

    @property
    def nedges(self):
        return int(self.edges.shape[1])

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (self.name + " mesh "
                "with " + str(self.p.shape[1]) + " vertices "
                "and " + str(self.t.shape[1]) + " elements.")

    @classmethod
    def dim(self):
        """Return the spatial dimension of the mesh."""
        return int(self.p.shape[0])

    def mapping(self):
        """Default local-to-global mapping for the mesh."""
        raise NotImplementedError("Default mapping not implemented!")

    def _uniform_refine(self):
        """Perform a single uniform mesh refinement."""
        raise NotImplementedError("Single refine not implemented "
                                  "for this mesh type!")

    def _adaptive_refine(self, marked):
        """Perform adaptive refinement."""
        raise NotImplementedError("Adaptive refine not implemented "
                                  "for this mesh type!")

    def refine(self: MeshType,
               arg: Optional[Union[int, ndarray]] = None) -> MeshType:
        """Refine the mesh.

        Parameters
        ----------
        arg
            Multiple variations: If None, refine all elements. If integer,
            perform multiple uniform refinements. If array of element
            indices, perform adaptive refinement.

        """
        if arg is None:
            self._uniform_refine()
        elif isinstance(arg, int):
            for itr in range(arg):
                self._uniform_refine()
        elif isinstance(arg, list):
            self._adaptive_refine(np.array(arg))
        elif isinstance(arg, ndarray):
            self._adaptive_refine(arg)
        else:
            raise NotImplementedError("The parameter type not supported.")
        return self

    def _fix_boundaries(self, facets: ndarray):
        """This should be called after each refine to update the indices in
        self.boundaries.

        Parameters
        ----------
        facets
            An array of integers of size no-splitted-elems x no-facets.

        """
        if hasattr(self, "boundaries") and self.boundaries is not None:
            for name in self.boundaries:
                self.boundaries[name] = (facets[:, self.boundaries[name]]
                                         .flatten())

    def remove_elements(self: MeshType, element_indices: ndarray) -> MeshType:
        """Construct new mesh with elements removed
        based on their indices.

        Parameters
        ----------
        element_indices
            List of element indices to remove.

        Returns
        -------
        Mesh
            A new mesh object with the requested elements removed.

        """
        warnings.warn("This method is deprecated in favour of prune",
                      DeprecationWarning)
        keep = np.setdiff1d(np.arange(self.t.shape[1]), element_indices)
        newt = self.t[:, keep]
        ptix = np.unique(newt)
        reverse = np.zeros(self.p.shape[1])
        reverse[ptix] = np.arange(len(ptix))
        newt = reverse[newt]
        newp = self.p[:, ptix]
        if newp.shape[1] == 0.0:
            raise Exception("The new mesh contains no points!")
        meshclass = type(self)
        return meshclass(newp, newt.astype(np.intp))

    def prune(self: MeshType, element_indices: ndarray) -> MeshType:
        """Construct new mesh with elements removed
        based on their indices.

        Parameters
        ----------
        element_indices
            List of element indices to remove.
        """
        keep = np.setdiff1d(np.arange(self.t.shape[1]), element_indices)
        p, t = self._reix(self.t[:, keep])
        meshclass = type(self)
        return meshclass(p, t)

    def scale(self: MeshType, scale: Union[float, DimTuple]) -> MeshType:
        """Scale the mesh.

        Parameters
        ----------
        scale
            Scale each dimension by this factor. If a single float is provided,
            same scaling is used for all dimensions. Otherwise, provide a
            tuple which has same size as the mesh dimension.

        """
        for itr in range(int(self.dim())):
            if isinstance(scale, tuple):
                self.p[itr, :] *= scale[itr]
            else:
                self.p[itr, :] *= scale
        return self

    def translate(self: MeshType, vec: DimTuple) -> MeshType:
        """Translate the mesh.

        Parameters
        ----------
        vec
            Translate the mesh by a vector. Must have same size as the mesh
            dimension.

        """
        for itr in range(int(self.dim())):
            self.p[itr, :] += vec[itr]
        return self

    def _validate(self):
        """Perform mesh validity checks."""
        # check that element connectivity contains integers
        # NOTE: this is necessary for some plotting functionality
        if not np.issubdtype(self.t[0, 0], np.integer):
            msg = ("Mesh._validate(): Element connectivity "
                   "must consist of integers.")
            raise Exception(msg)

        # check that vertex matrix has "correct" size
        if self.p.shape[0] > 3:
            msg = ("Mesh._validate(): We do not allow meshes "
                   "embedded into larger than 3-dimensional "
                   "Euclidean space! Please check that "
                   "the given vertex matrix is of size Ndim x Nvertices.")
            raise Exception(msg)

        # check that element connectivity matrix has correct size
        nvertices = {'line': 2, 'tri': 3, 'quad': 4, 'tet': 4, 'hex': 8}
        if self.t.shape[0] != nvertices[self.refdom]:
            msg = ("Mesh._validate(): The given connectivity "
                   "matrix has wrong shape!")
            raise Exception(msg)

        # check that there are no duplicate points
        tmp = np.ascontiguousarray(self.p.T)
        if self.p.shape[1] != np.unique(tmp.view([('', tmp.dtype)]
                                                 * tmp.shape[1])).shape[0]:
            msg = "Mesh._validate(): Mesh contains duplicate vertices."
            warnings.warn(msg)

        # check that all points are at least in some element
        if len(np.setdiff1d(np.arange(self.p.shape[1]),
                            np.unique(self.t))) > 0:
            msg = ("Mesh._validate(): Mesh contains a vertex "
                   "not belonging to any element.")
            raise Exception(msg)

    def expand_facets(self, facets: ndarray):
        """Find vertices and edges corresponding to given facets."""
        vertices = np.unique(self.facets[:, facets].flatten())
        edges = np.array([], dtype=np.int64)
        return vertices, edges

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
    def from_basis(cls: Type[MeshType], basis) -> MeshType:
        """Initialize a high-order mesh from :class:`skfem.assembly.Basis`."""
        if not isinstance(basis.mesh, cls):
            raise ValueError("Mesh and Basis must be compatible.")
        mesh = basis.mesh.copy()
        mesh.p = basis.doflocs
        mesh.t = basis.element_dofs
        return mesh

    @classmethod
    def load(cls: Type[MeshType], filename: str) -> MeshType:
        """Import a mesh from a file using `meshio
        <https://github.com/nschloe/meshio>`_.

        Parameters
        ----------
        filename
            The filename of the mesh.

        """
        from skfem.io.meshio import from_file
        return from_file(filename)

    def boundary_nodes(self) -> ndarray:
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:, self.boundary_facets()])

    def interior_nodes(self) -> ndarray:
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0, self.p.shape[1]),
                            self.boundary_nodes())

    def boundary_facets(self) -> ndarray:
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1, :] == -1)[0]

    def interior_facets(self) -> ndarray:
        """Return an array of interior facet indices."""
        return np.nonzero(self.f2t[1, :] >= 0)[0]

    def element_finder(self, mapping=None) -> Callable[[ndarray], ndarray]:
        """Return a function, which returns element
        indices corresponding to the input points."""
        raise NotImplementedError("element_finder not implemented "
                                  "for the given Mesh type.")

    def nodes_satisfying(self,
                         test: Callable[[ndarray], ndarray],
                         boundaries_only: bool = False) -> ndarray:
        """Return nodes that satisfy some condition.

        Parameters
        ----------
        test
            A function which returns True for the set of nodes that are to be
            included in the return set.
        boundaries_only
            If True, include only boundary facets.

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
            A function which returns True for the facet midpoints that are to
            be included in the return set.
        boundaries_only
            If True, include only boundary facets.

        """
        midp = [np.sum(self.p[itr, self.facets], axis=0) / self.facets.shape[0]
                for itr in range(self.p.shape[0])]
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
            A function which returns True for the element midpoints that are to
            be included in the return set.

        """
        midp = [np.sum(self.p[itr, self.t], axis=0) / self.t.shape[0]
                for itr in range(self.p.shape[0])]
        return np.nonzero(test(np.array(midp)))[0]

    @classmethod
    def from_dict(cls: Type[MeshType], d) -> MeshType:
        """Initialize a mesh from a dictionary."""
        if 'p' not in d or 't' not in d:
            raise ValueError("Dictionary must contain keys 'p' and 't'.")
        else:
            d['p'] = np.array(d['p']).T
            d['t'] = np.array(d['t']).T
        if 'boundaries' in d and d['boundaries'] is not None:
            d['boundaries'] = {k: np.array(v)
                               for k, v in d['boundaries'].items()}
        if 'subdomains' in d and d['subdomains'] is not None:
            d['subdomains'] = {k: np.array(v)
                               for k, v in d['subdomains'].items()}
        return cls(**d)

    def to_dict(self) -> Dict[str, ndarray]:
        """Return json serializable dictionary."""
        boundaries: Optional[Dict[str, ndarray]] = None
        subdomains: Optional[Dict[str, ndarray]] = None
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

    def define_boundary(self, name: str, test: Callable[[ndarray], ndarray],
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
        if self.boundaries is None:
            self.boundaries = {}
        self.boundaries[name] = self.facets_satisfying(test, boundaries_only)

    def _reix(self, ix: ndarray) -> Tuple[ndarray, ndarray]:
        """Connect ``self.p`` based on the indices ``ix``."""
        ixuniq = np.unique(ix)
        t = np.zeros(np.max(ix) + 1, dtype=np.int64)
        t[ixuniq] = np.arange(len(ixuniq), dtype=np.int64)
        return self.p[:, ixuniq], t[ix]

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

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
        boundaries = {
            **(self.boundaries if self.boundaries is not None else {}),
            **(other.boundaries if other.boundaries is not None else {}),
        } if self.boundaries is not None or other.boundaries is not None\
            else None
        subdomains = {
            **(self.subdomains if self.subdomains is not None else {}),
            **(other.subdomains if other.subdomains is not None else {}),
        } if self.subdomains is not None or other.subdomains is not None\
            else None
        return type(self)(p, t, boundaries, subdomains)

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p
