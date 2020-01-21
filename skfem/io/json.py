"""Import mesh from JSON as defined by :class:`skfem.mesh.to_dict`."""

import json

from skfem.mesh import *


def from_file(filename: str):
    with open(filename, 'r') as handle:
        d = json.load(handle)

    # detect dimension and number of vertices
    dim = len(d['p'][0])
    nverts = len(d['t'][0])

    if dim == 1:
        mesh_type = MeshLine
    elif dim == 2 and nverts == 3:
        mesh_type = MeshTri
    elif dim == 2 and nverts == 4:
        mesh_type = MeshQuad
    elif dim == 3 and nverts == 4:
        mesh_type = MeshTet
    elif dim == 3 and nverts == 8:
        mesh_type = MeshHex
    else:
        raise NotImplementedError("The given mesh is not supported.")

    return mesh_type.from_dict(d)


def to_file(mesh: Mesh, filename: str):
    with open(filename, 'w') as handle:
        d = json.dump(mesh.to_dict(), handle)
