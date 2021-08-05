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
    boundaries = {
        f"skfem:boundary:{name}": [
            (2 ** np.arange(mesh.t.shape[0])) @ np.isin(mesh.t2f, boundary)
        ]
        for name, boundary in mesh.boundaries.items()
    }
    cell_data = subdomains | boundaries
    return meshio.Mesh(
        mesh.p.T, [(TYPE_MESH_MAPPING[type(mesh)], mesh.t.T)], cell_data=cell_data
    )


meshes = Path(__file__).parent / "meshes"
for name, mesh in [
    ("disk", from_file(meshes / "disk.json")),
    ("beams", fe.Mesh.load(meshes / "beams.msh")),
]:
    meshio.write(Path(__file__).with_name(f"{name}.vtk"), to_meshio(mesh))
