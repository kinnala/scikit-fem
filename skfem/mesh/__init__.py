# -*- coding: utf-8 -*-
"""This module defines different types of finite element meshes.  Meshes can be
created using various built-in constructors or loaded from external formats
using `meshio <https://github.com/nschloe/meshio>`_.  The supported types are

- :class:`~skfem.mesh.MeshTri`, triangular mesh
- :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh
- :class:`~skfem.mesh.MeshTet`, tetrahedral mesh
- :class:`~skfem.mesh.MeshHex`, hexahedral mesh
- :class:`~skfem.mesh.MeshLine`, one-dimensional mesh

For example, initializing the default two triangle mesh for the unit square can
be done as follows:

>>> from skfem.mesh import MeshTri
>>> MeshTri()
Triangular mesh with 4 vertices and 2 elements.

"""

from .mesh import Mesh, MeshType
from .mesh_line import MeshLine
from .mesh2d import Mesh2D, MeshTri, MeshQuad
from .mesh3d import Mesh3D, MeshTet, MeshHex
