from skfem import *
from skfem.helpers import grad, dot
from skfem.io import from_meshio
from skfem.models.poisson import unit_load

from matplotlib.pyplot import subplots
import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

halfheight = 1.
length = 10.
thickness = halfheight

kratio = 80. / (4.181 / 7.14)

peclet = 357.


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


@BilinearForm
def conduction(u, v, w):
    return dot(w['w'] * grad(u), grad(v))


@BilinearForm
def advection(u, v, w):
    velocity_x = 1 - (w.x[1] / halfheight)**2  # plane Poiseuille
    return v * velocity_x * grad(u)[0]


conductivity = basis['heat'].zero_w() + 1
conductivity[mesh.subdomains['solid']] = kratio

longitudinal_gradient = 3 / 4 / peclet

A = (asm(conduction, basis['heat'], w=conductivity)
     + peclet * asm(advection, basis['fluid']))
b = (asm(unit_load, basis['heated'])
     + longitudinal_gradient
     * (asm(unit_load, basis['fluid-outlet'])
        + kratio * asm(unit_load, basis['solid-outlet'])))

D = basis['heat'].find_dofs(
    {label: boundary for
     label, boundary in mesh.boundaries.items()
     if label.endswith('-inlet')})
I = basis['heat'].complement_dofs(D)


def exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """return the exact fully developed solution at specified points"""
    return np.where(y > -halfheight,
                    - (5 - y**2) * (1 - y**2) / 16 - y / 2,
                    1 / 2 - (1 + y) / kratio) + longitudinal_gradient * x


temperature = np.zeros(basis['heat'].N)
inlet_dofs = basis['heat'].complement_dofs(I)
temperature[inlet_dofs] = exact(*mesh.p[:, inlet_dofs])

temperature = solve(*condense(A, b, temperature, I=I))

dofs = basis['heat'].find_dofs(
    {label: facets for label, facets in mesh.boundaries.items()
    if label.endswith('let')})

exit_interface_temperature = {
    'skfem': temperature[np.intersect1d(dofs['fluid-outlet'].all(),
                                        dofs['solid-outlet'].all())[0]],
    'exact': exact(length, -1.)
}

if __name__ == '__main__':
    from pathlib import Path
    from skfem.visuals.matplotlib import plot, savefig

    plot(mesh, temperature)
    savefig(Path(__file__).with_suffix('.png'),
            bbox_inches='tight', pad_inches=0)

    fig, ax = subplots()
    ax.set_title('transverse temperature profiles')

    y = {label: mesh.p[1, d.nodal['u']] for label, d in dofs.items()}
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

    ax.set_xlabel('temperature')
    ax.set_ylabel('$y$')
    ax.set_ylim((-halfheight - thickness, halfheight))
    ax.axhline(-halfheight, color='k', linestyle=':')
    ax.legend()
    fig.savefig(Path(__file__).with_name(
        Path(__file__).stem + '-inlet-outlet.png'))
