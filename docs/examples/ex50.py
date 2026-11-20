# Copyright (C) 2024 Radost Waszkiewicz and Jan Turczynowicz
# This software is published under BSD-3-clause license

# Folowing code solves advection-diffusion problem for
# temperature distribution around a cold sphere in warm
# liquid. Liquid flow is modelled using Stokes flow field.
# Thermal difusivity to advection ratio is controlled by
# Peclet number.

import numpy as np

from pathlib import Path
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm
from skfem import asm, solve, condense
from skfem.helpers import grad, dot

# Define the Peclet number
peclet = 30

#
# Code for creating mesh which we load
#

# floor_depth = 5.0
# floor_width = 5.0
# ball_size = 1.0
# ball_segments = 100
# mesh_size = 0.01
# far_mesh = 0.5

# box_points = [
#         ([0, -floor_depth], far_mesh),
#         ([floor_width, -floor_depth], far_mesh),
#         ([floor_width, floor_depth], far_mesh),
#         ([0, floor_depth], mesh_size),
#     ]

# phi_values = np.linspace(0, np.pi, ball_segments)
# ball_points = ball_size * np.column_stack((np.sin(phi_values), np.cos(phi_values)))
# mesh_boundary = np.vstack((
#     np.array([p for p,s  in box_points])
#     , ball_points))

# # Create the geometry and mesh using pygmsh
# with pygmsh.geo.Geometry() as geom:
#     poly = geom.add_polygon(
#         mesh_boundary,
#         mesh_size=([s for p,s in box_points]) + ([mesh_size] * len(ball_points)),
#     )

#     raw_mesh = geom.generate_mesh()

# # Convert the mesh to a skfem MeshTri object and define boundaries
# mesh = MeshTri(
#     raw_mesh.points[:, :2].T, raw_mesh.cells_dict["triangle"].T
# ).with_boundaries(
#     {
#         "left": lambda x: np.isclose(x[0], 0),  # Left boundary condition
#         "bottom": lambda x: np.isclose(x[1], -floor_depth),  # Bottom boundary condition
#         "ball": lambda x: x[0] ** 2 + x[1] ** 2 < 1.1 * ball_size**2,
#     }
# )

mesh = MeshTri.load(Path(__file__).parent / 'meshes' / 'cylinder_stokes.msh')

# Define the basis for the finite element method
basis = Basis(mesh, ElementTriP1())


@BilinearForm
def advection(u, v, p):
    """Advection bilinear form."""

    # Coordinate fields
    r, z = p.x

    us = 1  # velocity scale
    a = 1  # ball size

    # Stokes flow around a sphere of size `a`
    w = r ** 2 + z ** 2
    v_r = ((3 * a * r * z * us) / (4 * w ** 0.5)) * ((a / w) ** 2 - (1 / w))
    v_z = us + ((3 * a * us) / (4 * w ** 0.5)) * (
        (2 * a ** 2 + 3 * r ** 2) / (3 * w) - ((a * r) / w) ** 2 - 2
    )

    return (v_r * grad(u)[0] * v + v_z * grad(u)[1] * v) * 2 * np.pi * r


@BilinearForm
def claplace(u, v, p):
    """Laplace operator in cylindrical coordinates."""
    r = abs(p.x[0])
    return dot(grad(u), grad(v)) * 2 * np.pi * r


# Assemble the system matrix
A = asm(claplace, basis) + peclet * asm(advection, basis)

# Boundary condition
u = basis.zeros()
u[basis.get_dofs("bottom")] = 1.0
u[basis.get_dofs("ball")] = 0.0

u = solve(*condense(A, x=u, D=basis.get_dofs({"bottom", "ball"})))


def visualize():
    return basis.plot(u, 'web', cmap='jet')


if __name__ == "__main__":
    visualize().show()
