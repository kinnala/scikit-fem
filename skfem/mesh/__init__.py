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
<skfem MeshTri1 object>
  Number of elements: 2
  Number of vertices: 4
  Number of nodes: 4

Each mesh type has several constructors; see the docstring, e.g.,
``help(MeshTri)`` or click :class:`~skfem.mesh.MeshTri` in the online
documentation.  Importing from external formats can be done with the
constructor :meth:`~skfem.mesh.Mesh.load`.

"""
import numpy as np

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
from .mesh_quad_1_dg import MeshQuad1DG
from .mesh_hex_1_dg import MeshHex1DG
from .mesh_line_1_dg import MeshLine1DG
from .mesh_wedge_1 import MeshWedge1

# aliases
MeshTri = MeshTri1
MeshQuad = MeshQuad1
MeshTet = MeshTet1
MeshHex = MeshHex1


class MeshLineConstructor:

    def __call__(self, p=None, t=None, **kwargs):

        if p is not None:
            p = np.atleast_2d(p)

        if p is not None and t is None:
            tmp = np.arange(p.shape[1] - 1, dtype=np.int64)
            t = np.vstack((tmp, tmp + 1))
            return MeshLine1(p, t, **kwargs)

        if p is None and t is None:
            return MeshLine1(**kwargs)

        return MeshLine1(p, t, **kwargs)

    def __getattr__(self, name):
        return getattr(MeshLine1, name)


MeshLine = MeshLineConstructor()


__all__ = [
    "Mesh",
    "MeshLine",
    "MeshLine1",
    "MeshLine1DG",
    "Mesh2D",
    "MeshTri",
    "MeshTri1",
    "MeshTri2",
    "MeshTri1DG",
    "MeshQuad",
    "MeshQuad1",
    "MeshQuad2",
    "MeshQuad1DG",
    "Mesh3D",
    "MeshTet",
    "MeshTet1",
    "MeshTet2",
    "MeshHex",
    "MeshHex1",
    "MeshHex2",
    "MeshHex1DG",
    "MeshWedge1",
]
