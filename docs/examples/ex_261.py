from pathlib import Path

import numpy as np

import meshio
import skfem as fe
from skfem.io.json import from_file
from skfem.io.meshio import TYPE_MESH_MAPPING


def to_meshio(mesh: fe.Mesh) -> meshio.Mesh:
    cell_data = {
        f"skfem:subdomain:{name}": [
            np.isin(np.arange(mesh.t.shape[1]), subdomain).astype(int)
        ]
        for name, subdomain in mesh.subdomains.items()
    }
    return meshio.Mesh(
        mesh.p.T, [(TYPE_MESH_MAPPING[type(mesh)], mesh.t.T)], cell_data=cell_data
    )


mesh = from_file(Path(__file__).parent / "meshes" / "disk.json")
meshio.write(Path(__file__).with_suffix(".vtk"), to_meshio(mesh))
