# -*- coding: utf-8 -*-
"""This module defines different types of finite element meshes.  Meshes can
be created using various built-in constructors or loaded from external
formats using `meshio <https://github.com/nschloe/meshio>`_.  The supported
types are

- :class:`~skfem.mesh.MeshTri`, triangular mesh
- :class:`~skfem.mesh.MeshQuad`, quadrilateral mesh
- :class:`~skfem.mesh.MeshTet`, tetrahedral mesh
- :class:`~skfem.mesh.MeshHex`, hexahedral mesh
- :class:`~skfem.mesh.MeshLine`, one-dimensional mesh

For example, initializing the default mesh in the unit square can be done as
follows:

>>> from skfem.mesh import MeshTri
>>> MeshTri()
Triangular mesh with 4 vertices and 2 elements.

Each mesh type has several constructors, e.g.,

>>> MeshTri.init_lshaped()
Triangular mesh with 8 vertices and 6 elements.
>>> MeshTri.init_tensor([0.0, 1.0], [0.0, 1.0, 2.0])
Triangular mesh with 6 vertices and 4 elements.

A list of constructors can be found in the class docstring:

>>> help(MeshTri)

Importing from external formats can be done with the constructor
:meth:`~skfem.mesh.Mesh.load`.

"""

from .mesh import Mesh, MeshType
from .mesh_line import MeshLine
from .mesh2d import Mesh2D, MeshTri, MeshQuad, MeshTri2
from .mesh3d import Mesh3D, MeshTet, MeshHex


__all__ = [
    "Mesh",
    "MeshType",
    "MeshLine",
    "Mesh2D",
    "MeshTri",
    "MeshTri2",
    "MeshQuad",
    "Mesh3D",
    "MeshTet",
    "MeshHex"]
