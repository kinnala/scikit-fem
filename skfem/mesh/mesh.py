import numpy as np
import matplotlib.pyplot as plt
import warnings

from typing import Dict, Optional, Tuple,\
                   Type, TypeVar, Union,\
                   Callable
from numpy import ndarray

MeshType = TypeVar('MeshType', bound='Mesh')
DimTuple = Union[Tuple[float], Tuple[float, float], Tuple[float, float, float]]


class Mesh():
    """A finite element mesh.
    
    This is an abstract superclass. See the following implementations:

    - :class:`~skfem.mesh.MeshTri`, triangular mesh
    - :class:`~skfem.mesh.MeshTet`, tetrahedral mesh
    - :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh
    - :class:`~skfem.mesh.MeshHex`, hexahedral mesh
    - :class:`~skfem.mesh.MeshLine`, one-dimensional mesh

    """

    refdom: str = "none"  
    brefdom: str = "none" 
    meshio_type: str = "none"

    p: ndarray = np.array([]) 
    t: ndarray = np.array([]) 

    def __init__(self):
        """Check that p and t are C_CONTIGUOUS as this leads
        to better performance."""
        if self.p is not None:
            if self.p.flags['F_CONTIGUOUS']:
                if self.p.shape[1]>1000:
                    warnings.warn("Mesh.__init__(): Transforming " +
                            "over 100 vertices to C_CONTIGUOUS.")
                self.p = np.ascontiguousarray(self.p)
        if self.t is not None:
            if self.t.flags['F_CONTIGUOUS']:
                if self.t.shape[1]>1000:
                    warnings.warn("Mesh.__init__(): Transforming " +
                            "over 100 elements to C_CONTIGUOUS.")
                self.t = np.ascontiguousarray(self.t)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Mesh of type '" + str(type(self)) + "' "\
               "with " + str(self.p.shape) + " vertices " \
               "and " + str(self.t.shape) + " elements."

    def show(self):
        """A wrapper for matplotlib.pyplot.show()."""
        plt.show()

    def dim(self):
        """Return the spatial dimension of the mesh."""
        return int(self.p.shape[0])

    def mapping(self):
        """Default local-to-global mapping for the mesh."""
        raise NotImplementedError("Default mapping not implemented!")

    def _uniform_refine(self):
        """Perform a single uniform mesh refinement."""
        raise NotImplementedError("Single refine not implemented " +
                                  "for this mesh type!")

    def refine(self, no_refs: Optional[int] = None):
        """Refine the mesh.
        
        Parameters
        ----------
        no_refs
            Perform multiple refinements.

        """
        if no_refs is None:
            return self._uniform_refine()
        else:
            for itr in range(no_refs):
                self._uniform_refine()

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
        if newp.shape[1]==0.0:
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
        # NOTE: this is neccessary for some plotting functionality
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
        if len(np.setdiff1d(np.arange(self.p.shape[1]), np.unique(self.t))):
            msg = ("Mesh._validate(): Mesh contains a vertex "
                   "not belonging to any element.")
            raise Exception(msg)

    def save(self,
             filename: str,
             pointData: Optional[Union[ndarray, Dict[str, ndarray]]] = None,
             cellData: Optional[Union[ndarray, Dict[str, ndarray]]] = None) -> None:
        """Export the mesh and fields using meshio.

        Parameters
        ----------
        filename
            The filename for vtk-file.
        pointData
            Data related to the vertices of the mesh. Numpy array for one
            output or dict for multiple.
        cellData
            Data related to the elements of the mesh. Numpy array for one
            output or dict for multiple

        """
        import meshio

        if pointData is not None:
            if type(pointData) != dict:
                pointData = {'0':pointData}

        if cellData is not None:
            if type(cellData) != dict:
                cellData = {'0':cellData}

        cells = { self.meshio_type : self.t.T }
        mesh = meshio.Mesh(self.p.T, cells, pointData, cellData)
        meshio.write(filename, mesh)

    def _parse_submeshes(self) -> None:
        """Parse submeshes from self.external.

        Call after creating a mesh using Mesh.from_meshio to parse Mesh.external into
        Mesh.boundaries and Mesh.subdomains. Supports currently gmsh only.

        """

        # element to boundary element type mapping
        bnd_type = {
            'triangle':'line',
            'quad':'line',
            'tetra':'triangle',
            'hexahedron':'quad',
        }[self.meshio_type]

        def find_tagname(t):
            for key in self.external.field_data:
                if self.external.field_data[key][0] == t:
                    return key 

        # fill self.subdomains
        if self.meshio_type in self.external.cell_data and \
           'gmsh:physical' in self.external.cell_data[self.meshio_type]:
            elements = self.external.cells[self.meshio_type]
            elements_tag = self.external.cell_data[self.meshio_type]['gmsh:physical']
            
            self.subdomains = {}
            tags = np.unique(elements_tag)
            
            for tag in tags:
                t_set = np.nonzero(tag == elements_tag)[0]
                self.subdomains[find_tagname(tag)] = t_set

        # fill self.boundaries
        if bnd_type in self.external.cell_data and \
           'gmsh:physical' in self.external.cell_data[bnd_type]:
            facets = self.external.cells[bnd_type]
            facets_tag = self.external.cell_data[bnd_type]['gmsh:physical']
            bndfacets = self.boundary_facets()
            
            # put meshio facets to dict
            dic = {tuple(np.sort(facets[i])): facets_tag[i] for i in range(facets.shape[0])}
            
            # get index of corresponding Mesh.facets for each meshio
            # facet found in the dict
            ix = np.array([[dic[tuple(np.sort(self.facets[:, i]))], i]
                             for i in bndfacets
                             if tuple(np.sort(self.facets[:, i])) in dic])
            
            # read meshio tag numbers and names
            tags = ix[:, 0]
            self.boundaries = {}
                    
            for tag in np.unique(tags):
                tagix = np.nonzero(tags == tag)[0]
                self.boundaries[find_tagname(tag)] = ix[tagix, 1]
                
    @classmethod
    def from_meshio(cls: Type[MeshType], meshdata) -> MeshType:
        """Translate a mesh from `meshio
        <https://github.com/nschloe/meshio>`_.
        
        Parameters
        ----------
        meshdata
            A meshio.Mesh.

        Returns
        -------
        mesh
            The corresponding skfem.mesh object. The original meshio.Mesh
            object is accessible via the attribute mesh.external.

        """

        if cls.meshio_type in meshdata.cells:
            p = cls.strip_extra_coordinates(meshdata.points).T
            t = meshdata.cells[cls.meshio_type].T
            mesh = cls(p, t)
            mesh.external = meshdata
            
            # load submeshes, currently gmsh only
            try:
                mesh._parse_submeshes()
            except Exception as e:
                # all mesh formats are not supported; raise warning for
                # unsupported types
                warnings.warn("Unable to load submeshes.")
                print(e)

            return mesh
        else:
            raise Exception("The mesh contains no elements of type " + cls.meshio_type)

    @classmethod
    def load(cls: Type[MeshType], filename: str) -> MeshType:
        """Load an external mesh from file using `meshio
        <https://github.com/nschloe/meshio>`_.
        
        Parameters
        ----------
        filename
            The filename of the mesh.

        """
        import meshio
        return cls.from_meshio(meshio.read(filename))

    def boundary_nodes(self) -> ndarray:
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:, self.boundary_facets()])

    def interior_nodes(self) -> ndarray:
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0, self.p.shape[1]), self.boundary_nodes())

    def boundary_facets(self) -> ndarray:
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1, :] == -1)[0]

    def interior_facets(self) -> ndarray:
        """Return an array of interior facet indices."""
        return np.nonzero(self.f2t[1, :] >= 0)[0]

    def element_finder(self) -> Callable[[ndarray], ndarray]:
        """Return a function, which returns element
        indices corresponding to the input points."""
        raise NotImplementedError("element_finder not implemented" +\
                                  "for the given Mesh type.")
