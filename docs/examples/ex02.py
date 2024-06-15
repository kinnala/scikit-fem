r"""Kirchhoff plate problem.

This example demonstrates the solution a fourth order `Kirchhoff plate bending problem
<https://en.wikipedia.org/wiki/Kirchhoff%E2%80%93Love_plate_theory>`_ which
finds its applications in solid mechanics. For a stationary plate of constant
thickness :math:`d`, the governing equation reads: find the deflection :math:`u
: \Omega \rightarrow \mathbb{R}` that satisfies

.. math::
    \frac{Ed^3}{12(1-\nu^2)} \Delta^2 u = f \quad \text{in $\Omega$},
where :math:`\Omega = (0,1)^2`, :math:`f` is a perpendicular force,
:math:`E` and :math:`\nu` are material parameters.
In this example, we analyse a :math:`1\,\text{m}^2` plate of steel with thickness :math:`d=0.1\,\text{m}`.
The Young's modulus of steel is :math:`E = 200 \cdot 10^9\,\text{Pa}` and Poisson
ratio :math:`\nu = 0.3`.

There are several boundary conditions that the problem can take.
The *fully clamped* boundary condition reads

.. math::
    u = \frac{\partial u}{\partial \boldsymbol{n}} = 0,
where :math:`\boldsymbol{n}` is the outward normal.
Moreover, the *simply supported* boundary condition reads

.. math::
    u = 0, \quad M_{nn}(u)=0,
where :math:`M_{nn} = \boldsymbol{n} \cdot (\boldsymbol{M} \boldsymbol{n})`.
Finally, the *free* boundary condition reads

.. math::
    M_{nn}(u)=0, \quad V_{n}(u)=0,
where :math:`V_n` is the `Kirchhoff shear force <https://arxiv.org/pdf/1707.08396.pdf>`_. The exact
definition is not needed here as this boundary condition is a
natural one.

The correct weak formulation for the problem is: find :math:`u \in V` such that

.. math::
    \int_\Omega \boldsymbol{M}(u) : \boldsymbol{K}(v) \,\mathrm{d}x = \int_\Omega fv \,\mathrm{d}x \quad \forall v \in V,
where :math:`V` is now a subspace of :math:`H^2` with the essential boundary
conditions for :math:`u` and :math:`\frac{\partial u}{\partial \boldsymbol{n}}`.

Instead of constructing a subspace for :math:`H^2`, we discretise the problem
using the `non-conforming Morley finite element
<https://users.aalto.fi/~jakke74/WebFiles/Slides-Niiranen-ADMOS-09.pdf>`_ which
is a piecewise quadratic :math:`C^0`-continuous element for biharmonic problems.

"""
from skfem import *
from skfem.models.poisson import unit_load
from skfem.helpers import dd, ddot, trace, eye
import numpy as np

m = MeshTri.init_symmetric().refined(3)
basis = Basis(m, ElementTriMorley())

d = 0.1
E = 200e9
nu = 0.3


def C(T):
    return E / (1 + nu) * (T + nu / (1 - nu) * eye(trace(T), 2))


@BilinearForm
def bilinf(u, v, _):
    return d ** 3 / 12.0 * ddot(C(dd(u)), dd(v))


@LinearForm
def load(v, _):
    return 1e6 * v


K = bilinf.assemble(basis)
f = load.assemble(basis)

D = np.hstack((
    basis.get_dofs("left"),
    basis.get_dofs({"right", "top"}).all("u"),
))

x = solve(*condense(K, f, D=D))

def visualize():
    from skfem.visuals.matplotlib import draw, plot
    ax = draw(m)
    return plot(basis,
                x,
                ax=ax,
                shading='gouraud',
                colorbar=True,
                nrefs=2)

if __name__ == "__main__":
    visualize().show()
