from skfem import *
from skfem.importers import from_meshio
from skfem.models.poisson import unit_load

from matplotlib.pyplot import subplots
import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

halfheight = 0.1 / 2                  # mm
length = 1.
thickness = halfheight

volumetric_heat_capacity = 1. * 4181.  # uJ/uL.K

thermal_conductivity = {
    'fluid': 4.181 / 7.14,
    'wall': 80.
}                               # mW/mm.K

u0 = 1.                                          # mm/ms
heat_flux = 100. * thermal_conductivity['fluid']  # mW/mm**2


def make_mesh(halfheight: float,  # mm
              length: float,
              thickness: float) -> MeshTri:
    geom = Geometry()
    points = []
    lines = []

    lcar = halfheight / 2**3

    for xy in [(0., halfheight),
               (0., -halfheight),
               (length, -halfheight),
               (length, halfheight),
               (0., -halfheight - thickness),
               (length, -halfheight - thickness)]:
        points.append(geom.add_point([*xy, 0.], lcar))

    lines.append(geom.add_line(*points[:2]))
    geom.add_physical(lines[-1], 'fluid-inlet')

    lines.append(geom.add_line(*points[1:3]))

    lines.append(geom.add_line(*points[2:4]))
    outlet_lines = [lines[-1]]

    lines.append(geom.add_line(points[3], points[0]))

    geom.add_physical(geom.add_plane_surface(geom.add_line_loop(lines)),
                      'fluid')

    lines.append(geom.add_line(points[1], points[4]))
    geom.add_physical(lines[-1], 'solid-inlet')

    lines.append(geom.add_line(*points[4:6]))
    geom.add_physical(lines[-1], 'heated')

    lines.append(geom.add_line(points[5], points[2]))
    geom.add_physical(outlet_lines + lines[-1:], 'outlet')

    geom.add_physical(geom.add_plane_surface(geom.add_line_loop(
        [*lines[-3:], -lines[1]])), 'wall')

    return from_meshio(generate_mesh(geom, dim=2))


mesh = make_mesh(halfheight, length, thickness)
element = ElementTriP1()
basis = {
    'heat': InteriorBasis(mesh, element),
    'fluid': InteriorBasis(mesh, element, elements=mesh.subdomains['fluid']),
    'heating': FacetBasis(mesh, element, facets=mesh.boundaries['heated'])}


@bilinear_form
def conduction(u, du, v, dv, w):
    return w.w * sum(du * dv)


@bilinear_form
def advection(u, du, v, dv, w):
    _, y = w.x
    velocity_x = 1 - (y / halfheight)**2
    return v * velocity_x * du[0]


conductivity = basis['heat'].zero_w()
for subdomain, elements in mesh.subdomains.items():
    conductivity[elements] = thermal_conductivity[subdomain]

A = (asm(conduction, basis['heat'], w=conductivity)
     + volumetric_heat_capacity * u0 * asm(advection, basis['fluid']))
b = heat_flux * asm(unit_load, basis['heating'])

D = basis['heat'].get_dofs(
    {label: boundary for
     label, boundary in mesh.boundaries.items()
     if label.endswith('-inlet')})
I = basis['heat'].complement_dofs(D)

# prescribe exact fully developed solution on inlet plane
temperature = np.zeros(basis['heat'].N)
temperature[D['fluid-inlet'].all()] = heat_flux * (
    3 / 4 / halfheight / volumetric_heat_capacity
    * (mesh.p[0, D['fluid-inlet'].all()]
       - volumetric_heat_capacity * u0 / 12
       / thermal_conductivity['fluid'] / halfheight**2
       * (5 * halfheight**2 - mesh.p[1, D['fluid-inlet'].all()]**2)
       * (halfheight**2 - mesh.p[1, D['fluid-inlet'].all()]**2))
    - mesh.p[1, D['fluid-inlet'].all()] / 2 / thermal_conductivity['fluid'])
temperature[D['solid-inlet'].all()] = heat_flux * (
    3 / 4 / halfheight / volumetric_heat_capacity
    * mesh.p[0, D['solid-inlet'].all()]
    + halfheight / 2 / thermal_conductivity['fluid']
    - (halfheight + mesh.p[1, D['solid-inlet'].all()]) /
    thermal_conductivity['wall'])

temperature[I] = solve(*condense(A, b, temperature, I=I))

if __name__ == '__main__':

    from pathlib import Path

    mesh.plot(temperature, edgecolors='none')
    mesh.savefig(Path(__file__).with_suffix('.png'),
                 bbox_inches='tight', pad_inches=0)

    fig, ax = subplots()
    ax.set_title('transverse temperature profiles')
    outlet = basis['heat'].get_dofs(mesh.boundaries['outlet']).all()
    y = mesh.p[1, outlet]
    ii = np.argsort(y)
    ax.plot(temperature[outlet[ii]], y[ii], marker='o', label='outlet')
    inlet = basis['heat'].get_dofs(
        np.hstack([mesh.boundaries['fluid-inlet'],
                   mesh.boundaries['solid-inlet']])).all()
    y = mesh.p[1, inlet]
    ii = np.argsort(y)
    ax.plot(temperature[inlet[ii]], y[ii], marker='x', label='inlet')
    ax.set_xlabel('temperature / K')
    ax.set_ylabel('$y$ / mm')
    ax.set_ylim((-halfheight - thickness, halfheight))
    ax.legend()
    fig.savefig(Path(__file__).stem + '-inlet-outlet.png')
