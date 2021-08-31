"""Import mesh from JSON."""

import json
from os import PathLike
from typing import Type

import numpy as np

from skfem.mesh import MeshLine1, MeshTri, MeshQuad, MeshTet, MeshHex, Mesh


def to_dict(m):
    boundaries = None
    subdomains = None
    if m.boundaries is not None:
        boundaries = {k: v.tolist() for k, v in m.boundaries.items()}
    if m.subdomains is not None:
        subdomains = {k: v.tolist() for k, v in m.subdomains.items()}
    return {
        'p': m.p.T.tolist(),
        't': m.t.T.tolist(),
        'boundaries': boundaries,
        'subdomains': subdomains,
    }


def from_dict(cls, data):
    if 'p' not in data or 't' not in data:
        raise ValueError("Dictionary must contain keys 'p' and 't'.")
    else:
        data['p'] = np.ascontiguousarray(np.array(data['p']).T)
        data['t'] = np.ascontiguousarray(np.array(data['t']).T)
    if 'boundaries' in data and data['boundaries'] is not None:
        data['boundaries'] = {k: np.array(v)
                              for k, v in data['boundaries'].items()}
    if 'subdomains' in data and data['subdomains'] is not None:
        data['subdomains'] = {k: np.array(v)
                              for k, v in data['subdomains'].items()}
    data['doflocs'] = data.pop('p')
    data['_subdomains'] = data.pop('subdomains')
    data['_boundaries'] = data.pop('boundaries')
    return cls(**data)


def from_file(filename: PathLike) -> Mesh:
    with open(filename, 'r') as handle:
        d = json.load(handle)

    # detect dimension and number of vertices
    dim = len(d['p'][0])
    nverts = len(d['t'][0])

    mesh_type: Type = Mesh

    if dim == 1:
        mesh_type = MeshLine1
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

    return from_dict(mesh_type, d)


def to_file(mesh: Mesh, filename: str):
    with open(filename, 'w') as handle:
        json.dump(mesh.to_dict(), handle)
