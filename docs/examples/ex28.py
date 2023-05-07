r"""Conjugate heat transfer.

The forced convection example can be extended to conjugate heat transfer by
giving a finite thickness and thermal conductivity to one of the walls.

The example is modified to a configuration for which there is a fully developed
solution which can be found in closed form: given a uniform (but possibly
different) heat flux over each of the walls, the temperature field
asymptotically is the superposition of a uniform longitudinal gradient and a
transverse profile; here the analysis of a pipe of circular section (Bird,
Stewart, & Lightfoot 1960, §3–13) is modified for a planar duct (Mallinson,
McBain, & Brown 2019).  The two-dimensional equation in the fluid :math:`-1 < y
< 1`

.. math::
   \mathrm{Pe} \left(1 - y^2\right) \frac{\partial T}{\partial x} =
   \nabla^2 T, \qquad (-1 < y < 1)
has the exact solution

.. math::
   T(x, y) = \frac{3x}{4\mathrm{Pe}} - \frac{(5-y^2)(1-y^2)}{16} - \frac{y}{2}.
Here Pe is the dimensionless Péclet number based on the centreline-velocity and channel half-height.

The governing equation in the solid is :math:`\nabla \cdot k\nabla T = 0` where
:math:`k` the ratio of the thermal conductivity to that of the fluid.  This has
the exact solution matching that in the fluid along the interface
:math:`y = -1` is:

.. math::
   T (x, y) = \frac{3x}{4\mathrm{Pe}} +  \frac{1}{2} - \frac{1+y}{k},
   \qquad (y < -1)

This combined fully developed solution applies throughout the entire domain if
it is specified along the fluid and solid inlet :math:`x = 0`, if the ceiling
is insulated :math:`\partial T/\partial y = 0` on :math:`y = 1`, if the floor
is uniformly heated :math:`k\partial t/\partial y = 1` on :math:`y = -2`, and
if the longitudinal gradient is applied as a uniform Neumann condition on the
outlet, :math:`k\partial T/\partial x = 3k/4\mathrm{Pe}` on :math:`x = \ell`.

In conjugate heat transfer, part of the domain is advection–conduction while the
rest is pure conduction; the main difficulty is specifying different governing
equations on different subdomains.  One way to do this has already been
demonstrated: :ref:`subdomain` by only assembling the operator over a basis
restricted to the elements belonging to a particular subdomain.

* Bird, R. B., W. E. Stewart, & E. N. Lightfoot (1960). Transport Phenomena. New York: Wiley
* Mallinson, S. G., McBain, G. D. & Brown, B. R. (2019). `Conjugate
  heat transfer in thermal inkjet printheads
  <https://www.researchgate.net/publication/334028691_Conjugate_Heat_Transfer_in_Thermal_Inkjet_Printheads>`_.
  14th International Conference on Heat Transfer, Fluid Mechanics and
  Thermodynamics, Wicklow, Ireland.

"""
from packaging import version
from pathlib import Path

from skfem import *
from skfem.helpers import grad, dot
from skfem.models.poisson import unit_load

import numpy as np


halfheight = 1.
length = 10.
thickness = halfheight

kratio = 80. / (4.181 / 7.14)

peclet = 357.

mesh = Mesh.load(Path(__file__).parent / 'meshes' / 'ex28.msh')
element = ElementTriP1()
basis = {
    'heat': Basis(mesh, element),
    'fluid': Basis(mesh, element, elements=mesh.subdomains['fluid']),
    **{label: FacetBasis(mesh, element, facets=mesh.boundaries[label])
       for label in ['heated', 'fluid-outlet', 'solid-outlet']}}


@BilinearForm
def conduction(u, v, w):
    return dot(w['conductivity'] * grad(u), grad(v))


@BilinearForm
def advection(u, v, w):
    velocity_x = 1 - (w.x[1] / halfheight)**2  # plane Poiseuille
    return v * velocity_x * grad(u)[0]


basis0 = basis['heat'].with_element(ElementTriP0())
conductivity = basis0.zeros() + 1
conductivity[mesh.subdomains['solid']] = kratio

longitudinal_gradient = 3 / 4 / peclet

A = (asm(conduction,
         basis['heat'],
         conductivity=basis0.interpolate(conductivity))
     + peclet * asm(advection, basis['fluid']))
b = (asm(unit_load, basis['heated'])
     + longitudinal_gradient
     * (asm(unit_load, basis['fluid-outlet'])
        + kratio * asm(unit_load, basis['solid-outlet'])))

D = basis["heat"].get_dofs([b for b in mesh.boundaries if b.endswith("-inlet")])
I = basis['heat'].complement_dofs(D)


def exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """return the exact fully developed solution at specified points"""
    return np.where(y > -halfheight,
                    - (5 - y**2) * (1 - y**2) / 16 - y / 2,
                    1 / 2 - (1 + y) / kratio) + longitudinal_gradient * x


temperature = basis['heat'].zeros()
inlet_dofs = basis['heat'].complement_dofs(I)
temperature[inlet_dofs] = exact(*mesh.p[:, inlet_dofs])

temperature = solve(*condense(A, b, temperature, I=I))

exit_interface_temperature = {
    "skfem": temperature[
        np.intersect1d(
            *(
                basis["heat"].get_dofs(b)
                for b in mesh.boundaries
                if b.endswith("-outlet")
            )
        )
    ][0],
    "exact": exact(length, -1.0),
}

if __name__ == '__main__':
    from pathlib import Path
    from skfem.visuals.matplotlib import plot, savefig
    from matplotlib.pyplot import subplots

    plot(mesh, temperature)
    savefig(Path(__file__).with_suffix('.png'),
            bbox_inches='tight', pad_inches=0)

    fig, ax = subplots()
    ax.set_title('transverse temperature profiles')

    dofs = basis['heat'].get_dofs(mesh.boundaries)
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
            ax.plot(temperature[dofs[label].nodal_ix[ii[label]]], y[label][ii[label]],
                    marker=marker, color=color, linestyle='none',
                    label=f'{label}, skfem')
        ax.plot(exact(mesh.p[0, dofs[label].nodal_ix[0]], y['exact']), y['exact'],
                color='k', linestyle=linestyle, label=f'{port}, exact')

    ax.set_xlabel('temperature')
    ax.set_ylabel('$y$')
    ax.set_ylim((-halfheight - thickness, halfheight))
    ax.axhline(-halfheight, color='k', linestyle=':')
    ax.legend()
    fig.savefig(Path(__file__).with_name(
        Path(__file__).stem + '-inlet-outlet.png'))
