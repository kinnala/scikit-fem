"""Computation of the characteristic impedance and velocity factor of RG316
coaxial cable.

.. note::
   This example requires the external package
   `pygmsh <https://pypi.org/project/pygmsh/>`_.

This example solves the series inductance (per meter) and parallel capacitance
(per meter)
of RG316 coaxial cable. These values are then used to compute the
characteristic impedance and velocity factor of the cable.

From transmission line theory it is known, that the characteristic impedance
of a lossless transmission line is

.. math::
   Z = \\sqrt{\\frac{L}{C}},

while the phase velocity is

.. math::
   v = \\frac{1}{\\sqrt{L C}},

where :math:`L` is the series inductance per unit length of the transmission
line and :math:`C` is the parallel capacitance per unit length of the
transmission line.

Further, the phase velocity relative to the speed of light is called the
velocity factor of the transmission line.

RG316
-----

A piece of coaxial cable is composed of an inner conductor, which is
surrounded by a dielectric insulator.
The dielectric insulator in turn is surrounded by an outer conductor. Finally,
the outer conductor is surrounded by an outer insulator.

For an RG316 cable, the dimensions and materials of the components are

- Inner conductor: OD 0.5mm, silver plated copper
- Inner insulator: OD 1.52mm, PTFE
- Outer conductor: OD 1.98mm, silver plated copper
- Outer insulator: OD 2.48mm, PEF

RG316 has a nominal characteristic impedance of 50 ohms and a velocity factor
of 0.69.

Inductance
----------

Inductance of the cable is computed using the magnetostatic equations

.. math::
   \\nabla \\cdot \\boldsymbol{B} = 0

   \\nabla \\times \\boldsymbol{H} = \\boldsymbol{J}

and the constitutive relation

.. math::
   \\boldsymbol{B} = \\mu \\boldsymbol{H},

where :math:`\\mu` is the permability of the medium.

Since :math:`\\boldsymbol{B}` is divergence free, it can be written in terms
of a vector potential :math:`\\boldsymbol{A}` as

.. math::
   \\boldsymbol{B} = \\nabla \\times \\boldsymbol{A}.

Thus we have the strong form for the vector potential :math:`\\boldsymbol{A}`
as

.. math::
   \\nabla \\times (\\mu^{-1} \\nabla \\times \\boldsymbol{A}) =
   \\boldsymbol{J}.

The corresponding weak form is: find :math:`\\boldsymbol{A} \\in V` such that

.. math::
   \\int_{\\Omega}
   (\\mu^{-1} \\nabla \\times \\boldsymbol{A}) \\cdot
   (\\nabla \\times \\boldsymbol{v})\\ \\mathrm{d}x -
   \\int_{\\partial \\Omega}
   (\\mu^{-1} \\nabla \\times \\boldsymbol{A}) \\times
   \\boldsymbol{n} \\cdot \\boldsymbol{v}\\ \\mathrm{d}x =
   \\int_{\\Omega} \\boldsymbol{J} \\cdot \\boldsymbol{v}\\ \\mathrm{d}x
   \\quad \\forall \\boldsymbol{v} \\in V.

We take the boundary condition
:math:`\\boldsymbol{B} \\cdot \\boldsymbol{n} = 0` on
:math:`\\partial \\Omega`, which is equivalent to
:math:`\\boldsymbol{A} = 0` on
:math:`\\partial \\Omega`. This is an essential boundary condition, which is
enforced by the choice of :math:`V`.
Thus we have the final weak form: find :math:`\\boldsymbol{A} \\in V` such that

.. math::
   \\int_{\\Omega}
   (\\mu^{-1} \\nabla \\times \\boldsymbol{A}) \\cdot
   (\\nabla \\times \\boldsymbol{v})\\ \\mathrm{d}x =
   \\int_{\\Omega} \\boldsymbol{J} \\cdot \\boldsymbol{v}\\ \\mathrm{d}x
   \\quad \\forall \\boldsymbol{v} \\in V.


For this application :math:`\\Omega` is taken to be the cross section of the
coaxial cable, and it is assumed that the cable has infinite length. It is
assumed that the functions :math:`\\boldsymbol{J}`, :math:`\\boldsymbol{A}` as
well as and any :math:`\\boldsymbol{v} \\in V` depend only on coordinates
:math:`x_1` and :math:`x_2` in the cross-section plane, and have a nonzero
component only in the perpendicular direction to the cross-section plane. In
other words, they are assumed to have the following form

.. math::
   \\boldsymbol{J} &= J(x_1, x_2) \\boldsymbol{e_3}

   \\boldsymbol{A} &= A(x_1, x_2) \\boldsymbol{e_3}

   \\boldsymbol{v} &= v(x_1, x_2) \\boldsymbol{e_3}

This reduces the problem to two dimensions. Taking the curl of a function of
the assumed form and substituting the result in the weak form gives a more
familiar weak form in the cross-section plane: find :math:`A \\in V` such that

.. math::
   \\int_{\\Omega}
   \\mu^{-1} (\\nabla A \\cdot \\nabla v)\\ \\mathrm{d}x =
   \\int_{\\Omega} J v\\ \\mathrm{d}x \\quad \\forall v \\in V.


In order to actually compute the inductance (per unit length) of the cable, a
current is passed through the inner conductor while an equal but opposite
current is passed through the outer conductor. The energy (per unit length)
stored in the produced magnetic field is computed as

.. math::
   E = \\frac{1}{2} \\int_{\\Omega} \\mu^{-1} |\\nabla A|^2\\ \\mathrm{d}x

However, the energy (per unit length) stored in the magnetic field of the
inductor can also be stated in terms of its inductance (per unit length) as

.. math::
   E = \\frac{1}{2} L I^2,

where :math:`L` is the inductance (per unit length) and :math:`I` is the
current passed through the inductor. Thus

.. math::
   L = \\frac{2 E}{I^2}


Capacitance
-----------

Capacitance of the cable is computed using the electrostatic equations

.. math::
   \\nabla \\times \\boldsymbol{E} = \\boldsymbol{0}

   \\nabla \\cdot \\boldsymbol{D} = 0

and the constitutive relation

.. math::
   \\boldsymbol{D} = \\epsilon \\boldsymbol{E},

where :math:`\\epsilon` is the permittivity of the medium.

Since :math:`\\boldsymbol{E}` is curl-free, it can be written in terms of a
scalar potential :math:`U` as

.. math::
   \\boldsymbol{E} = -\\nabla U

Thus we have the strong form for the scalar potential :math:`U` as

.. math::
   -\\nabla \\cdot (\\epsilon \\nabla U) = 0.

However, this equation is only meaningful in a dielectric medium. In a
conductor, the electric field is zero, and thus the potential is constant (and
conceptually :math:`\\epsilon \\rightarrow \\infty`). The conductors need to
be excluded from the computation domain.

In any case, the equation has the familiar weak form: find :math:`U` such that

.. math::
   \\int_{\\Omega} \\epsilon \\nabla U \\cdot \\nabla v\\ \\mathrm{d}x = 0
   \\quad \\forall v \\in V.

Consider again the cross-section plane of the coaxial cable, with the cable
itself extending out-of-plane into infinity. Take :math:`U` to depend only on
the coordinates :math:`x_1` and :math:`x_2` in the cross-section plane. This
again reduces the problem to two dimensions.

Due to conducting media in the cross-section, the problem needs to split into
two domains: the first domain consisting of the inner insulator and the second
domain consisting of the space outside the outer conductor. In both domains,
we have a non-homogeneous Dirichlet boundary condition for :math:`U` on the
conductor surfaces, while in the second domain the potential has a homogeneous
Neumann condition on the free space boundary.

In order to actually compute the capacitance (per unit length) of the cable, a
potential is set on the inner conductor while an equal but opposite potential
is set on the outer conductor. The energy (per unit length) of the produced
electic field is computed as

.. math::
   E = \\frac{1}{2} \\int_{\\Omega} \\epsilon | \\nabla U |^2\\ \\mathrm{d}x

However, the energy (per unit length) stored in the electic field of a
capacitor can also be stated in terms of its capacitance (per unit length) as

.. math::
   E = \\frac{1}{2} C V^2,

where :math:`C` is the capacitance (per unit length) and :math:`V` is the
potential difference across the capacitor. Thus

.. math::
   C = \\frac{2 E}{V^2}.

License
-------

Copyright 2018-2020 scikit-fem developers

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
from packaging import version

from skfem.mesh import MeshTri
from skfem.assembly import InteriorBasis, FacetBasis
from skfem.utils import solve, asm, condense, project
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace, unit_load, mass

import pygmsh
if version.parse(pygmsh.__version__) < version.parse('7.0.0'):
    from pygmsh.built_in import Geometry
else:
    from pygmsh.geo import Geometry
from skfem.io import from_meshio

import numpy as np


def make_mesh() -> MeshTri:
    # dimensions for RG316 coaxial cable
    inner_conductor_diameter = 0.50e-3
    inner_insulator_diameter = 1.52e-3
    outer_conductor_diameter = 1.98e-3
    outer_insulator_diameter = 2.48e-3

    # characteristic length for mesh generation
    lcar = 0.1e-3

    geom = Geometry()

    inner_conductor = geom.add_circle(
        (0, 0, 0), inner_conductor_diameter/2,
        lcar=lcar)
    geom.add_physical(
        inner_conductor.plane_surface, label='inner_conductor')
    geom.add_physical(
        inner_conductor.line_loop.lines, label='inner_conductor_outer_surface')

    inner_insulator = geom.add_circle(
        (0, 0, 0), inner_insulator_diameter/2,
        lcar=lcar, holes=[inner_conductor.line_loop])
    geom.add_physical(
        inner_insulator.plane_surface, label='inner_insulator')
    geom.add_physical(
        inner_insulator.line_loop.lines, label='outer_conductor_inner_surface')

    outer_conductor = geom.add_circle(
        (0, 0, 0), outer_conductor_diameter/2,
        lcar=lcar, holes=[inner_insulator.line_loop])
    geom.add_physical(
        outer_conductor.plane_surface, label='outer_conductor')
    geom.add_physical(
        outer_conductor.line_loop.lines, label='outer_conductor_outer_surface')

    outer_insulator = geom.add_circle(
        (0, 0, 0), outer_insulator_diameter/2,
        lcar=lcar, holes=[outer_conductor.line_loop])
    geom.add_physical(
        outer_insulator.plane_surface, label='outer_insulator')
    geom.add_physical(
        outer_insulator.line_loop.lines, label='boundary')

    if version.parse(pygmsh.__version__) < version.parse('7.0.0'):
        return from_meshio(pygmsh.generate_mesh(geom, dim=2))
    else:
        return from_meshio(geom.generate_mesh(dim=2))


mesh = make_mesh()

element = ElementTriP1()

# permeability of vacuum
mu0 = 1.25663706212e-6
# permittivity of vacuum
eps0 = 8.8541878128e-12

# relative permittivity of polytetrafluoroethylene
eps_ptfe = 2.1
# relative permittivity of fluorinated ethylene propylene
eps_fep = 2.1


global_basis = InteriorBasis(mesh, element)
inner_conductor_basis = InteriorBasis(
    mesh, element, elements=mesh.subdomains['inner_conductor'])
outer_conductor_basis = InteriorBasis(
    mesh, element, elements=mesh.subdomains['outer_conductor'])
inner_insulator_basis = InteriorBasis(
    mesh, element, elements=mesh.subdomains['inner_insulator'])
outer_insulator_basis = InteriorBasis(
    mesh, element, elements=mesh.subdomains['outer_insulator'])

inner_conductor_outer_surface_basis = FacetBasis(
    mesh, element, facets=mesh.boundaries['inner_conductor_outer_surface'])
outer_conductor_inner_surface_basis = FacetBasis(
    mesh, element, facets=mesh.boundaries['outer_conductor_inner_surface'])

dofs = {
    'boundary':
    global_basis.get_dofs(mesh.boundaries['boundary']),
    'inner_conductor_outer_surface':
    global_basis.get_dofs(mesh.boundaries['inner_conductor_outer_surface']),
    'outer_conductor_inner_surface':
    global_basis.get_dofs(mesh.boundaries['outer_conductor_inner_surface'])
}

# functional to compute the integral of a load vector over the domain
load_integral = solve(asm(mass, global_basis), asm(unit_load, global_basis))

# all materials have a relative permeability of effectively 1
K_mag = asm(laplace, global_basis) * (1/mu0)

# pass 1A through the conductors
current = 1

J_inner_conductor = asm(unit_load, inner_conductor_basis)
# scale inner conductor current density to have an integral
# equal to current over domain
J_inner_conductor *= current / np.dot(J_inner_conductor, load_integral)

J_outer_conductor = asm(unit_load, outer_conductor_basis)
# scale outer conductor current density to have an integral
# equal to -current over domain
J_outer_conductor *= -current / np.dot(J_outer_conductor, load_integral)

A = solve(*condense(
    K_mag, J_inner_conductor + J_outer_conductor, D=dofs['boundary']))

# magnetic field energy from FEM
E_mag = 0.5*np.dot(A, K_mag*A)
# energy stored in inductor: E = 0.5*L*I^2
# thus, L = 2*E/I^2
L = 2*E_mag / (current**2)

print(f'L={L} H/m')

# assemble the parts of the stiffness matrix for each material separately
K_elec_inner_insulator = asm(laplace, inner_insulator_basis) * eps0 * eps_ptfe
K_elec_outer_insulator = asm(laplace, outer_insulator_basis) * eps0 * eps_fep
# use dummy value for permittivity, uniform U in conductor
K_elec_inner_conductor = asm(laplace, inner_conductor_basis) * eps0
# use dummy value for permittivity, uniform U in conductor
K_elec_outer_conductor = asm(laplace, outer_conductor_basis) * eps0

# global stiffness matrix is the sum of the subdomain contributions
K_elec = (
    K_elec_inner_insulator + K_elec_outer_insulator +
    K_elec_inner_conductor + K_elec_outer_conductor)

# set a 1V potential difference between the conductors
voltage = 1

# initialize the non-homogeneous Dirichlet conditions on the conductor surfaces
U = np.zeros(K_elec.shape[0])
U[dofs['inner_conductor_outer_surface'].all()] = project(
    lambda x: voltage/2, basis_to=inner_conductor_outer_surface_basis,
    I=dofs['inner_conductor_outer_surface'])
U[dofs['outer_conductor_inner_surface'].all()] = project(
    lambda x: -voltage/2, basis_to=outer_conductor_inner_surface_basis,
    I=dofs['outer_conductor_inner_surface'])

U = solve(*condense(
    K_elec, np.zeros(K_elec.shape[1]), U,
    D=dofs['inner_conductor_outer_surface'] |
    dofs['outer_conductor_inner_surface']))

# electric field energy
E_elec = 0.5*np.dot(U, K_elec*U)
# energy stored in a capacitor: E = 0.5*C*U^2
# thus, C = 2*E/(U^2)
C = 2*E_elec/(voltage**2)

print(f'C={C} F/m')

# characteristic impedance of coax
Z = np.sqrt(L/C)
print(f'Z={Z} ohm')

# phase velocity factor (fraction of speed of light)
v = (1/np.sqrt(L*C)) / 299792458

print(f'v={v} c')

if __name__ == '__main__':
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import plot, savefig
    import matplotlib.pyplot as plt
    from skfem.utils import derivative

    B_x = derivative(A, global_basis, global_basis, 1)
    B_y = -derivative(A, global_basis, global_basis, 0)

    E_x = -derivative(U, global_basis, global_basis, 0)
    E_y = -derivative(U, global_basis, global_basis, 1)

    fig = plt.figure(figsize=(11.52, 5.12))

    ax1 = plt.subplot(1, 2, 1)
    plot(global_basis, np.sqrt(B_x**2 + B_y**2), ax=ax1, colorbar=True)
    ax1.set_title('Magnetic flux density (Tesla)')
    ax1.set_aspect('equal')
    ax1.set_yticks([])

    ax2 = plt.subplot(1, 2, 2)
    plot(global_basis, np.sqrt(E_x**2 + E_y**2), ax=ax2, colorbar=True)
    ax2.set_title('Electric field strength (V/m)')
    ax2.set_aspect('equal')
    ax2.set_yticks([])

    savefig(splitext(argv[0])[0] + '_solution.png')
