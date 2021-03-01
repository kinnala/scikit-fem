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

from .base_mesh import BaseMesh as Mesh
from .base_mesh import MeshLine1 as MeshLine
from .base_mesh import BaseMesh3D as Mesh3D
from .base_mesh import MeshTet1 as MeshTet
from .base_mesh import MeshHex1 as MeshHex
from .base_mesh import BaseMesh2D as Mesh2D
from .base_mesh import MeshTri1 as MeshTri
from .base_mesh import MeshQuad1 as MeshQuad
from .base_mesh import (MeshTri1, MeshTri2, MeshQuad1, MeshQuad2, MeshTet1,
                        MeshHex1, MeshLine1)


__all__ = [
    "Mesh",
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
