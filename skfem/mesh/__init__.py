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

from .mesh import Mesh
from .mesh_2d import Mesh2D
from .mesh_3d import Mesh3D
from .mesh_hex_1 import MeshHex1
from .mesh_hex_2 import MeshHex2
from .mesh_line_1 import MeshLine1
from .mesh_quad_1 import MeshQuad1
from .mesh_quad_2 import MeshQuad2
from .mesh_tet_1 import MeshTet1
from .mesh_tet_2 import MeshTet2
from .mesh_tri_1 import MeshTri1
from .mesh_tri_2 import MeshTri2
from .mesh_tri_1_dg import MeshTri1DG

# aliases
MeshLine = MeshLine1
MeshTri = MeshTri1
MeshQuad = MeshQuad1
MeshTet = MeshTet1
MeshHex = MeshHex1


__all__ = [
    "Mesh",
    "MeshLine",
    "Mesh2D",
    "MeshTri",
    "MeshTri1",
    "MeshTri2",
    "MeshTri1DG",
    "MeshQuad",
    "MeshQuad1",
    "MeshQuad2",
    "Mesh3D",
    "MeshTet",
    "MeshTet1",
    "MeshTet2",
    "MeshHex",
    "MeshHex1",
    "MeshHex2",
]
