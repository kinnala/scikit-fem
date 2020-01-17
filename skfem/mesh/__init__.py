# -*- coding: utf-8 -*-
"""This module contains different types of finite element meshes.

Meshes are created using various built-in constructors or loaded from external
formats using `meshio <https://github.com/nschloe/meshio>`_. See the following
implementations:

- :class:`~skfem.mesh.MeshTri`, triangular mesh
- :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh
- :class:`~skfem.mesh.MeshTet`, tetrahedral mesh
- :class:`~skfem.mesh.MeshHex`, hexahedral mesh
- :class:`~skfem.mesh.MeshLine`, one-dimensional mesh

"""

from .mesh import Mesh, MeshType
from .mesh_line import MeshLine
from .mesh2d import Mesh2D, MeshTri, MeshQuad
from .mesh3d import Mesh3D, MeshTet, MeshHex
