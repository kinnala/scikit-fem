# Copyright (C) 2024 Radost Waszkiewicz and Jan Turczynowicz
# This software is published under BSD-3-clause license

# Folowing code solves advection-diffusion problem for
# temperature distribution around a cold sphere in warm
# liquid. Liquid flow is modelled using Stokes flow field.
# Thermal difusivity to advection ratio is controlled by
# Peclet number.

import pygmsh
import numpy as np
from skfem import *
from skfem.models.poisson import laplace
from skfem.visuals.matplotlib import plot, show
import matplotlib.pyplot as plt
from skfem.helpers import grad, dot

# Define the Peclet number
peclet = 30

floor_depth = 5.0
floor_width = 5.0
ball_size = 1.0
ball_segments = 100
mesh_size = 0.01
far_mesh = 0.5

#
# Code for creating mesh which we load
#

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

mesh = MeshTri.load("./meshes/cylinder_stokes.msh")

# Define the basis for the finite element method
basis = Basis(mesh, ElementTriP1())

left_nodes = mesh.p[:, basis.get_dofs("left").flatten()]
bottom_nodes = mesh.p[:, basis.get_dofs("bottom").flatten()]
ball_nodes = mesh.p[:, basis.get_dofs("ball").flatten()]

# Draw the mesh
mesh.draw()
plt.plot(left_nodes[0], left_nodes[1], "x")
plt.plot(bottom_nodes[0], bottom_nodes[1], "x")
plt.plot(ball_nodes[0], ball_nodes[1], "o")
show()


@BilinearForm
def advection(u, v, w):
    """Advection bilinear form."""

    # Coordinate fields
    r, z = w.x

    U = 1  # velocity scale
    a = 1  # ball size

    squared_dist = r**2 + z**2
    v_r = ((3 * a * r * z * U) / (4 * (squared_dist) ** 0.5)) * (
        (a / (squared_dist)) ** 2 - (1 / (squared_dist))
    )
    v_z = U + ((3 * a * U) / (4 * (squared_dist) ** 0.5)) * (
        (2 * a**2 + 3 * r**2) / (3 * (squared_dist))
        - ((a * r) / (squared_dist)) ** 2
        - 2
    )

    advection_velocity_x = v_r
    advection_velocity_y = v_z
    return (
        (v * advection_velocity_x * grad(u)[0] + v * advection_velocity_y * grad(u)[1])
        * 2
        * np.pi
        * r
    )


@BilinearForm
def claplace(u, v, w):
    """Laplace operator in cylindrical coords."""
    r = abs(w.x[1])
    return dot(grad(u), grad(v)) * 2 * np.pi * r


# Identify the interior degrees of freedom
interior = basis.complement_dofs(basis.get_dofs({"bottom", "ball"}))

# Assemble the system matrix
A = asm(claplace, basis) + peclet * asm(advection, basis)

# Initialize the solution vector with boundary conditions
u = basis.zeros()

u[basis.get_dofs("bottom")] = 1.0  # Bottom boundary condition
u[basis.get_dofs("ball")] = 0.0  # Bottom boundary condition

# Solve the system
u = solve(*condense(A, x=u, I=interior))

if __name__ == "__main__":
    # Plot the solution

    # Define the parameters
    U = 1.0  # Example value for U
    R = 1.0  # Example value for R

    # mesh.draw()
    r = np.linspace(0, 5, 100)
    z = np.linspace(-5, 5, 100)
    R_grid, Z_grid = np.meshgrid(r, z)

    plt.tripcolor(mesh.p[0], mesh.p[1], mesh.t.T, u, shading="gouraud", cmap="viridis")
    plt.colorbar()
    plt.clim(vmin=0, vmax=1)  # Set color range
    plt.gca().set_aspect(
        "equal", "box"
    )  # 'equal' ensures that one unit in x is equal to one unit in y
    plt.tight_layout()

    plt.show()
