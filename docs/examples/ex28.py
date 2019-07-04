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
    'solid': 80.
}                               # mW/mm.K

u0 = 1.                                          # mm/ms
heat_flux = 100. * thermal_conductivity['fluid']  # mW/mm**2


def make_mesh(halfheight: float,  # mm
              length: float,
              thickness: float) -> MeshTri:
    geom = Geometry()
    points = []
    lines = []

    lcar = halfheight / 2**2

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
    geom.add_physical(lines[-1], 'fluid-outlet')

    lines.append(geom.add_line(points[3], points[0]))

    geom.add_physical(geom.add_plane_surface(geom.add_line_loop(lines)),
                      'fluid')

    lines.append(geom.add_line(points[1], points[4]))
    geom.add_physical(lines[-1], 'solid-inlet')

    lines.append(geom.add_line(*points[4:6]))
    geom.add_physical(lines[-1], 'heated')

    lines.append(geom.add_line(points[5], points[2]))
    geom.add_physical(lines[-1], 'solid-outlet')

    geom.add_physical(geom.add_plane_surface(geom.add_line_loop(
        [*lines[-3:], -lines[1]])), 'solid')

    return from_meshio(generate_mesh(geom, dim=2))


mesh = make_mesh(halfheight, length, thickness)
element = ElementTriP1()
basis = {
    'heat': InteriorBasis(mesh, element),
    'fluid': InteriorBasis(mesh, element, elements=mesh.subdomains['fluid']),
    **{label: FacetBasis(mesh, element, facets=mesh.boundaries[label])
       for label in ['heated', 'fluid-outlet', 'solid-outlet']}}


@bilinear_form
def conduction(u, du, v, dv, w):
    return w.w * sum(du * dv)


@bilinear_form
def advection(u, du, v, dv, w):
    _, y = w.x
    velocity_x = 1 - (y / halfheight)**2  # plane Poiseuille
    return v * velocity_x * du[0]


conductivity = basis['heat'].zero_w()
for subdomain, elements in mesh.subdomains.items():
    conductivity[elements] = thermal_conductivity[subdomain]

longitudinal_gradient = (3 * heat_flux
                         / 4 / volumetric_heat_capacity / halfheight / u0)

A = (asm(conduction, basis['heat'], w=conductivity)
     + volumetric_heat_capacity * u0 * asm(advection, basis['fluid']))
b = (heat_flux * asm(unit_load, basis['heated'])
     + longitudinal_gradient * (
         thermal_conductivity['fluid'] * asm(unit_load, basis['fluid-outlet'])
         + thermal_conductivity['solid'] * asm(unit_load, basis['solid-outlet'])
     ))

D = basis['heat'].get_dofs(
    {label: boundary for
     label, boundary in mesh.boundaries.items()
     if label.endswith('-inlet')})
I = basis['heat'].complement_dofs(D)


def exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """return the exact fully developed solution at specified points"""
    return heat_flux * np.where(
        y > -halfheight,
        3 / 4 / halfheight / volumetric_heat_capacity
        * (x
           - volumetric_heat_capacity * u0 / 12
           / thermal_conductivity['fluid'] / halfheight**2
           * (5 * halfheight**2 - y**2) * (halfheight**2 - y**2))
        - y / 2 / thermal_conductivity['fluid'],
        3 * x / 4 / volumetric_heat_capacity / halfheight / u0
        + halfheight / 2 / thermal_conductivity['fluid']
        - (halfheight + y) / thermal_conductivity['solid'])

temperature = np.zeros(basis['heat'].N)
inlet_dofs = basis['heat'].complement_dofs(I)
temperature[inlet_dofs] = exact(*mesh.p[:, inlet_dofs])

temperature[I] = solve(*condense(A, b, temperature, I=I))

if __name__ == '__main__':

    from pathlib import Path

    mesh.plot(temperature, edgecolors='none')
    mesh.savefig(Path(__file__).with_suffix('.png'),
                 bbox_inches='tight', pad_inches=0)

    fig, ax = subplots()
    ax.set_title('transverse temperature profiles')
    
    dofs = {label: basis['heat'].get_dofs(facets).all()
            for label, facets in mesh.boundaries.items()
            if label.endswith('let')}
    y = {label: mesh.p[1, d] for label, d in dofs.items()}
    ii = {label: np.argsort(yy) for label, yy in y.items()}

    y['exact'] = np.linspace(min(y['solid-inlet']),
                             max(y['fluid-inlet']))
    for port, saturation, linestyle in [('inlet', '', '--'),
                                        ('outlet', 'dark', '-')]:
        for phase, hue, marker in [('fluid', 'green', 'x'),
                                   ('solid', 'red', '+')]:
            color = saturation + hue
            label = f'{phase}-{port}'
            ax.plot(temperature[dofs[label][ii[label]]], y[label][ii[label]],
                    marker=marker, color=color, linestyle='none',
                    label=f'{label}, skfem')
        ax.plot(exact(mesh.p[0, dofs[label][0]], y['exact']), y['exact'],
                color='k', linestyle=linestyle, label=f'{port}, exact')
    
    ax.set_xlabel('temperature / K')
    ax.set_ylabel('$y$ / mm')
    ax.set_ylim((-halfheight - thickness, halfheight))
    ax.axhline(-halfheight, color='k', linestyle=':')
    ax.legend()
    fig.savefig(Path(__file__).with_name(
        Path(__file__).stem + '-inlet-outlet.png'))
