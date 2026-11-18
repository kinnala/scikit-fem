r"""Elastic beam with embedded rebars.

This example models a cantilever beam with reinforcement bars (rebars)
embedded in a three-dimensional elastic domain.  The bars are described as
one-dimensional bars along a set of mesh edges, i.e. a path tagged using
:meth:`~skfem.mesh.Mesh.with_paths`.  The bars are perfectly bonded to the
surrounding material, i.e. they share the displacement field :math:`u`, and
resist only axial deformation.  The additional stiffness of the bars is

.. math::

   \int_\Gamma EA\,\varepsilon(u)\,\varepsilon(v)\,\mathrm{d}s,
   \quad \varepsilon(u) = t \cdot (\nabla u)\, t,

where :math:`\Gamma` is the path, :math:`EA` is the axial stiffness of a bar
and :math:`t` is the unit tangent of the path.  It is assembled using
:class:`~skfem.assembly.EdgeBasis` which integrates over the edges of the mesh
and provides the tangent as ``w.t``.

The beam is clamped at :math:`x=0` and loaded downwards at the free end.  The
bars are placed on the top side of the beam which is in tension.  Away from
the ends, the axial force of the bars agrees with the beam theory for the
transformed cross section.  The Poisson ratio is zero to allow for a direct
comparison with the beam theory.

"""
import numpy as np
from skfem import *
from skfem.helpers import ddot, sym_grad, dot, mul, eye, trace
from skfem.models.elasticity import lame_parameters

L, b, h = 5., 1., 1.  # dimensions of the beam
E, nu = 1., 0.  # material parameters of the beam
EA = .05  # axial stiffness of a bar
zr = .8  # height of the bars
F = 1e-3  # load at the free end

m = (
    MeshTet.init_tensor(np.linspace(0, L, 11),
                        np.array([0., .25, .75, 1.]),
                        np.array([0., .4, zr, 1.]))
    .with_paths({
        'rebar': lambda x: (np.isclose(x[2], zr)
                            & (np.isclose(x[1], .25)
                               | np.isclose(x[1], .75))),
    })
    .with_boundaries({
        'clamp': lambda x: np.isclose(x[0], 0.),
        'end': lambda x: np.isclose(x[0], L),
    })
)

e = ElementVector(ElementTetP2())
basis = Basis(m, e)
rebar = EdgeBasis(m, e, edges='rebar')
lam, mu = lame_parameters(E, nu)


@BilinearForm
def elasticity(u, v, w):
    def C(T):
        return 2. * mu * T + lam * eye(trace(T), 3)
    return ddot(C(sym_grad(u)), sym_grad(v))


def axial_strain(u, t):
    return dot(t, mul(u.grad, t))


@BilinearForm
def bars(u, v, w):
    return EA * axial_strain(u, w.t) * axial_strain(v, w.t)


@LinearForm
def load(v, w):
    return -F / (b * h) * v[2]


K = elasticity.assemble(basis)
f = load.assemble(basis.boundary('end'))
D = basis.get_dofs('clamp')

u0 = solve(*condense(K, f, D=D))  # without the bars
u = solve(*condense(K + bars.assemble(rebar), f, D=D))

# vertical displacement at the center of the free end
probe = basis.probes(np.array([[L], [b / 2], [h / 2]]))
deflection0 = (probe @ u0)[2]
deflection = (probe @ u)[2]

# axial force of the bars at the quadrature points
x = rebar.global_coordinates()[0].ravel()
N = (EA * axial_strain(rebar.interpolate(u), rebar.tangents)).ravel()

# beam theory for the transformed cross section of two bars
zc = (b * h * h / 2 + 2 * EA / E * zr) / (b * h + 2 * EA / E)
I_eff = (b * h ** 3 / 12 + b * h * (h / 2 - zc) ** 2
         + 2 * EA / E * (zr - zc) ** 2)


def N_beam(x):
    return EA * F * (L - x) * (zr - zc) / (E * I_eff)


def visualize():
    import matplotlib.pyplot as plt
    ix = np.argsort(x)
    xs = np.linspace(0, L, 100)
    fig, ax = plt.subplots()
    ax.plot(x[ix], N[ix], 'k.', label='finite element')
    ax.plot(xs, N_beam(xs), 'r-', label='beam theory')
    ax.set_xlabel('$x$')
    ax.set_ylabel('axial force of a bar')
    ax.legend()
    return ax


if __name__ == '__main__':
    print("Deflection without bars: {:.5f}".format(deflection0))
    print("Deflection with bars: {:.5f}".format(deflection))
    visualize()
    import matplotlib.pyplot as plt
    plt.show()
