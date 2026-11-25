r"""Plate with a circular hole between rigid grips.

This example computes the stiffness of the square plate
:math:`(-1, 1)^2` with a circular hole of radius :math:`1/2` at its
centre.  The plate is linear elastic (plane stress, :math:`E = 1`,
:math:`\nu = 0.3`) and is pulled by two frictionless grips: the faces
:math:`x = \pm 1` are moved apart by :math:`\delta = 1` in the normal
direction while the shear traction on them vanishes.  The edges
:math:`y = \pm 1` and the hole are traction free.  By symmetry, only the
quarter :math:`x, y > 0` is modelled, with :math:`u_x = 0` on :math:`x = 0`
and :math:`u_y = 0` on :math:`y = 0`.

The quadratic quadrilateral mesh is built directly in the code: the nodes of
a structured grid, including the edge midpoints and the element centres, are
mapped onto the quarter plate so that the elements next to the hole follow
the circle.  The displacement is approximated with biquadratic elements.
Only :math:`u_x` is prescribed on the grip, and the Dirichlet conditions are
applied by :func:`~skfem.utils.condense`.  The reaction force :math:`F` on
the grip is the sum of the assembled residual :math:`Au` over the
constrained DOFs of the grip, which gives the stiffness :math:`F/\delta` of
the whole plate.  Finally, the average of :math:`\sigma_{xx}` over the small
patch :math:`\tfrac12 \leq r \leq 0.55`, :math:`|\theta - \pi/2| \leq 0.1`
at the edge of the hole is evaluated using
:class:`~skfem.assembly.Functional` on a basis restricted to the elements of
the patch.

With this coarse mesh the stiffness lies within 0.5 % of the interval
:math:`0.5912009869 \leq F/\delta \leq 0.5912009882` certified in the
benchmark pack at
https://github.com/simulationstation/CertiField/tree/main/benchpack.

"""
import numpy as np
from skfem import *
from skfem.helpers import sym_grad
from skfem.models.elasticity import (linear_elasticity, linear_stress,
                                     plane_stress)

a = .5  # radius of the hole
rp = .55  # outer radius of the patch
dth = .1  # half opening angle of the patch
delta = 1.  # distance by which the grips are moved apart
lam, mu = plane_stress(1., .3)


def to_plate(th, s):
    """Map the parameters (th, s) to the quarter plate.

    The ring 0 <= s <= 1 maps to the annulus a <= r <= rp and 1 <= s <= 1 + m
    blends from the circle r = rp to the outer edges x = 1 (th < pi/4) and
    y = 1 (th > pi/4).

    """
    c, d = np.cos(th), np.sin(th)
    ring = a + (rp - a) * np.minimum(s, 1.)
    edge = np.where(th < np.pi / 4,
                    np.array([np.ones_like(th), np.tan(th)]),
                    np.array([1. / np.tan(np.maximum(th, np.pi / 4)),
                              np.ones_like(th)]))
    t = np.maximum(s - 1., 0.) / m
    return (1. - t) * ring * np.array([c, d]) + t * edge


# element boundaries in the angular and the radial directions; the grid has
# lines at th = pi/4 (the corner of the plate) and th = pi/2 - dth (the edge
# of the patch)
th = np.concatenate((np.linspace(0., np.pi / 4, 4),
                     np.linspace(np.pi / 4, np.pi / 2 - dth, 4)[1:],
                     [np.pi / 2]))
m = 4
s = np.arange(m + 2.)

# all nodes of the biquadratic elements: element boundaries and midpoints
TH = np.repeat(np.interp(np.arange(2 * len(th) - 1) / 2,
                         np.arange(len(th)), th), 2 * len(s) - 1)
S = np.tile(np.arange(2 * len(s) - 1) / 2, 2 * len(th) - 1)
p = to_plate(TH, S)


def node(i, j):
    return i * (2 * len(s) - 1) + j


# corners (counterclockwise), edge midpoints and centre of each element
i, j = np.meshgrid(2 * np.arange(len(th) - 1), 2 * np.arange(len(s) - 1),
                   indexing='ij')
i, j = i.ravel(), j.ravel()
t = np.array([node(i, j), node(i, j + 2), node(i + 2, j + 2), node(i + 2, j),
              node(i, j + 1), node(i + 1, j + 2), node(i + 2, j + 1),
              node(i + 1, j), node(i + 1, j + 1)])

mesh = MeshQuad2(p, t).with_boundaries({
    'grip': lambda x: np.isclose(x[0], 1.),
    'symmetry_x': lambda x: np.isclose(x[0], 0.),
    'symmetry_y': lambda x: np.isclose(x[1], 0.),
})

e = ElementVector(ElementQuad2())
basis = Basis(mesh, e)
A = linear_elasticity(lam, mu).assemble(basis)

grip = basis.get_dofs('grip').all('u^1')
D = np.concatenate((grip,
                    basis.get_dofs('symmetry_x').all('u^1'),
                    basis.get_dofs('symmetry_y').all('u^2')))
u = basis.zeros()
u[grip] = delta / 2
u = solve(*condense(A, x=u, D=D))

# the reaction force on the grip of the quarter is half of the whole grip
force = 2. * (A @ u)[grip].sum()
stiffness = force / delta

# the elements of the ring with their centre at th > pi/2 - dth form the patch
centre = t[8]
in_patch = (S[centre] < 1.) & (TH[centre] > np.pi / 2 - dth)
patch = Basis(mesh, e, elements=np.nonzero(in_patch)[0])


@Functional
def sigma_xx(w):
    return linear_stress(lam, mu)(sym_grad(w['u']))[0, 0]


sigma_xx_avg = sigma_xx.assemble(patch, u=u) / patch.dx.sum()


def visualize():
    from skfem.visuals.matplotlib import draw, plot
    sbasis = basis.with_element(ElementQuad2())
    s_xx = sbasis.project(
        linear_stress(lam, mu)(sym_grad(basis.interpolate(u)))[0, 0]
    )
    ax = plot(sbasis, s_xx, shading='gouraud', nrefs=3, colorbar=True)
    draw(mesh, ax=ax)
    ax.set_title(r'$\sigma_{xx}$')
    return ax


if __name__ == '__main__':
    print("Stiffness: {:.6f}".format(stiffness))
    print("Average of sigma_xx over the patch: {:.6f}".format(sigma_xx_avg))
    visualize()
    import matplotlib.pyplot as plt
    plt.show()
