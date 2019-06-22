import warnings
from typing import Dict, Optional, Tuple,\
                   Type, TypeVar, Union,\
                   Callable, Any

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


MeshType = TypeVar('MeshType', bound='Mesh')
DimTuple = Union[Tuple[float],
                 Tuple[float, float],
                 Tuple[float, float, float]]


class Mesh():
    """A finite element mesh.

    This is an abstract superclass. See the following implementations:

    - :class:`~skfem.mesh.MeshTri`, triangular mesh
    - :class:`~skfem.mesh.MeshTet`, tetrahedral mesh
    - :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh
    - :class:`~skfem.mesh.MeshHex`, hexahedral mesh
    - :class:`~skfem.mesh.MeshLine`, one-dimensional mesh

    Attributes
    ----------
    p
        The vertices of the mesh (dim x Nvertices). Each column corresponds to
        a point.
    t
        The element connectivity (dim x Nelements). Each column corresponds to
        a element and contains four column indices to p.
    refdom
        A string describing the shape of the reference domain. Used to find
        quadrature rules.
    brefdom
        A string describing the shape of the reference domain for element
        boundaries. Used for finding quadrature rules.
    meshio_type
        A string which is used to convert between scikit-fem and meshio mesh
        types.
    name
        A string which is used in pretty printing the object.
    subdomains
        Named subsets of elements.
    boundaries
        Named subsets of boundary facets.
    external
        If Mesh is loaded from external format (object), the original
        representation is kept here.

    """

    refdom: str = "none"
    brefdom: str = "none"
    meshio_type: str = "none"
    name: str = "Abstract"

    p: ndarray = np.array([])
    t: ndarray = np.array([])

    subdomains: Optional[Dict[str, ndarray]] = None
    boundaries: Optional[Dict[str, ndarray]] = None
    external: Any = None

    def __init__(self):
        """Check that p and t are C_CONTIGUOUS as this leads
        to better performance."""
        if self.p is not None:
            if not isinstance(self.p, ndarray):
                self.p = np.array(self.p, dtype=np.float_)
            if self.p.flags['F_CONTIGUOUS']:
                if self.p.shape[1] > 1000:
                    warnings.warn("Mesh.__init__(): Transforming "
                                  "over 100 vertices to C_CONTIGUOUS.")
                self.p = np.ascontiguousarray(self.p)
        if self.t is not None:
            if not isinstance(self.t, ndarray):
                self.t = np.array(self.t, dtype=np.intp)
            if self.t.flags['F_CONTIGUOUS']:
                if self.t.shape[1] > 1000:
                    warnings.warn("Mesh.__init__(): Transforming "
                                  "over 100 elements to C_CONTIGUOUS.")
                self.t = np.ascontiguousarray(self.t)
        # transform lists to ndarrays
        if self.boundaries is not None:
            for k, v in self.boundaries.items():
                if not isinstance(v, ndarray):
                    self.boundaries[k] = np.array(v, dtype=np.intp)
        if self.subdomains is not None:
            for k, v in self.subdomains.items():
                if not isinstance(v, ndarray):
                    self.subdomains[k] = np.array(v, dtype=np.intp)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (self.name + " mesh "
                "with " + str(self.p.shape[1]) + " vertices "
                "and " + str(self.t.shape[1]) + " elements.")

    def show(self):
        """A wrapper for matplotlib.pyplot.show()."""
        plt.show()

    def savefig(self, *args, **kwargs):
        """A wrapper for matplotlib.pyplot.savefig()."""
        plt.savefig(*args, **kwargs)

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

    def refine(self, arg: Optional[Union[int, ndarray]] = None):
        """Refine the mesh.

        Parameters
        ----------
        arg
            Multiple variations:
            - If None, refine all elements.
            - If integer, perform multiple uniform refinements.
            - If array of element indices, adaptively refine.

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

    def remove_elements(self, element_indices: ndarray) -> MeshType:
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

    def scale(self, scale: Union[float, DimTuple]) -> None:
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

    def translate(self, vec: DimTuple) -> None:
        """Translate the mesh.

        Parameters
        ----------
        vec
            Translate the mesh by a vector. Must have same size as the mesh
            dimension.

        """
        for itr in range(int(self.dim())):
            self.p[itr, :] += vec[itr]

    def _validate(self):
        """Perform mesh validity checks."""
        # check that element connectivity contains integers
        # NOTE: this is necessary for some plotting functionality
        if not np.issubdtype(self.t[0, 0], np.signedinteger):
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

    def save(self,
             filename: str,
             point_data: Optional[Dict[str, ndarray]] = None,
             cell_data: Optional[Dict[str, ndarray]] = None) -> None:
        """Export the mesh and fields using meshio.

        Parameters
        ----------
        filename
            The output filename, with suffix determining format;
            e.g. .msh, .vtk, .xdmf
        point_data
            Data related to the vertices of the mesh.
        cell_data
            Data related to the elements of the mesh.

        """
        import meshio

        if point_data is not None:
            if not isinstance(point_data, dict):
                raise ValueError("point_data should be "
                                 "a dictionary of ndarrays.")

        if cell_data is not None:
            if not isinstance(point_data, dict):
                raise ValueError("cell_data should be "
                                 "a dictionary of ndarrays.")

        cells = {self.meshio_type: self.t.T}
        mesh = meshio.Mesh(self.p.T, cells, point_data, cell_data)
        meshio.write(filename, mesh)

    @classmethod
    def load(cls: Type[MeshType], filename: str) -> MeshType:
        """Import a mesh from file using `meshio
        <https://github.com/nschloe/meshio>`_.

        Parameters
        ----------
        filename
            The filename of the mesh.

        """
        from skfem.importers.meshio import from_file
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

    def element_finder(self) -> Callable[[ndarray], ndarray]:
        """Return a function, which returns element
        indices corresponding to the input points."""
        raise NotImplementedError("element_finder not implemented "
                                  "for the given Mesh type.")

    def nodes_satisfying(self, test: Callable[[ndarray], bool]) -> ndarray:
        """Return nodes that satisfy some condition.

        Parameters
        ----------
        test
            A function which returns True for the set of nodes that are to be
            included in the return set.

        """
        return np.nonzero(test(self.p))[0]

    def facets_satisfying(self, test: Callable[[ndarray], bool]) -> ndarray:
        """Return facets whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            A function which returns True for the facet midpoints that are to
            be included in the return set.

        """
        midp = [np.sum(self.p[itr, self.facets], axis=0) / self.facets.shape[0]
                for itr in range(self.p.shape[0])]
        return np.nonzero(test(np.array(midp)))[0]

    def elements_satisfying(self,
                            test: Callable[[ndarray], bool]) -> ndarray:
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

    def to_dict(self) -> Dict[str, ndarray]:
        """Return json serializable dictionary."""
        if self.boundaries is not None:
            boundaries = {k: v.tolist() for k, v in self.boundaries.items()}
        else:
            boundaries = self.boundaries
        if self.subdomains is not None:
            subdomains = {k: v.tolist() for k, v in self.subdomains.items()}
        else:
            subdomains = self.subdomains
        return {
            'p': self.p.tolist(),
            't': self.t.tolist(),
            'boundaries': boundaries,
            'subdomains': subdomains,
        }

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        return p
