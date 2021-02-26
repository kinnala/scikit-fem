"""This module defines finite element meshes.  Meshes can be created using
built-in constructors or loaded from external formats using `meshio
<https://github.com/nschloe/meshio>`_.  The supported types are

- :class:`~skfem.mesh.MeshTri`, triangular mesh
- :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh
- :class:`~skfem.mesh.MeshTet`, tetrahedral mesh
- :class:`~skfem.mesh.MeshHex`, hexahedral mesh
- :class:`~skfem.mesh.MeshLine`, one-dimensional mesh

Default constructor creates a mesh for the unit square:

>>> from skfem.mesh import MeshTri
>>> MeshTri()
Triangular mesh with 4 vertices and 2 elements.

Each mesh type has several constructors; see the docstring, e.g.,
``help(MeshTri)`` or click :class:`~skfem.mesh.MeshTri` in the online
documentation.  Importing from external formats can be done with the
constructor :meth:`~skfem.mesh.Mesh.load`.

"""

from .mesh import Mesh, MeshType
from .mesh_line import MeshLine
from .mesh2d import Mesh2D, MeshTri, MeshQuad
from .mesh3d import Mesh3D, MeshTet, MeshHex
from .base_mesh import (MeshTri1, MeshTri2, MeshQuad1, MeshQuad2, MeshTet1,
                        MeshHex1)


__all__ = [
    "Mesh",
    "MeshType",
    "MeshLine",
    "Mesh2D",
    "MeshTri",
    "MeshTri1",
    "MeshTri2",
    "MeshQuad",
    "MeshQuad1",
    "MeshQuad2",
    "Mesh3D",
    "MeshTet",
    "MeshTet1",
    "MeshHex",
    "MeshHex1",
]
