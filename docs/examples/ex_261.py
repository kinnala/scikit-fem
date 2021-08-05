from pathlib import Path

import numpy as np

import meshio
import skfem as fe
from skfem.io.json import from_file
from skfem.io.meshio import TYPE_MESH_MAPPING


def to_meshio(mesh: fe.Mesh) -> meshio.Mesh:
    subdomains = {
        f"skfem:subdomain:{name}": [
            np.isin(np.arange(mesh.t.shape[1]), subdomain).astype(int)
        ]
        for name, subdomain in mesh.subdomains.items()
    }
    boundaries = {}
    for name, boundary in mesh.boundaries.items():
        name = f"skfem:boundary:{name}"
        b = np.isin(mesh.t2f, boundary)
        if b.sum(0).max() > 1:  # a cell has more than one facet on boundary
            raise NotImplementedError
        boundaries[name] = [np.full(mesh.t.shape[1:], mesh.t.shape[0])]  # sentinel
        bmask = np.nonzero(b)
        boundaries[name][0][bmask[1]] = bmask[0]

    cell_data = subdomains | boundaries
    return meshio.Mesh(
        mesh.p.T, [(TYPE_MESH_MAPPING[type(mesh)], mesh.t.T)], cell_data=cell_data
    )


mesh = from_file(Path(__file__).parent / "meshes" / "disk.json")
meshio.write(Path(__file__).with_suffix(".vtk"), to_meshio(mesh))
