from pathlib import Path

import numpy as np

import meshio
import skfem as fe
from skfem.io.json import from_file
from skfem.io.meshio import MESH_TYPE_MAPPING, TYPE_MESH_MAPPING


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


def from_meshio(mio: meshio.Mesh) -> fe.Mesh:
    p = mio.points.T
    t = mio.cells[0].data.T
    mesh_type = MESH_TYPE_MAPPING[mio.cells[0].type]
    mtmp = mesh_type(p, t)
    boundaries = {}
    subdomains = {}
    for name, data in mio.cell_data.items():
        subnames = name.split(":")
        if subnames[0] != "skfem":
            continue
        if subnames[1] == "subdomain":
            subdomains[subnames[2]] = np.nonzero(data[0])[0]
        elif subnames[1] == "boundary":
            boundaries[subnames[2]] = mtmp.t2f[
                (2 ** np.arange(t.shape[0]))[:, None] & data[0] > 0
            ]
    return mesh_type(p, t, boundaries, subdomains)


meshes = Path(__file__).parent / "meshes"
for name, mesh in [
    ("disk", from_file(meshes / "disk.json")),
    ("beams", fe.Mesh.load(meshes / "beams.msh")),
]:
    if name == "disk":
        mesh.boundaries["interface"] = np.logical_and(
            *[np.logical_or(*np.isin(mesh.f2t, s)) for s in mesh.subdomains.values()]
        )
    mio = to_meshio(mesh)
    meshio.write(Path(__file__).with_name(f"{name}.vtk"), mio)
    mesh2 = from_meshio(mio)
    print("\n", name)
    print("As read:", mesh)
    print("To meshio:", mio)
    print("Reconverted:", mesh2, "\n")
    for subdomain_name, subdomain in mesh.subdomains.items():
        assert np.setxor1d(subdomain, mesh2.subdomains[subdomain_name]).size == 0
        print(subdomain_name)
    for boundary_name, boundary in mesh.boundaries.items():
        assert np.setxor1d(boundary, mesh2.boundaries[boundary_name]).size == 0
        print(boundary_name)
