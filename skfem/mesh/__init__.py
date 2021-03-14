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
from .mesh import Mesh2D
from .mesh import Mesh3D
from .mesh import MeshLine1 as MeshLine
from .mesh import MeshTri1 as MeshTri
from .mesh import MeshQuad1 as MeshQuad
from .mesh import MeshTet1 as MeshTet
from .mesh import MeshHex1 as MeshHex
from .mesh import MeshTri2, MeshQuad2, MeshTet2, MeshHex2


__all__ = [
    "Mesh",
    "MeshLine",
    "Mesh2D",
    "MeshTri",
    "MeshTri2",
    "MeshQuad",
    "MeshQuad2",
    "Mesh3D",
    "MeshTet",
    "MeshTet2",
    "MeshHex",
    "MeshHex2",
]
