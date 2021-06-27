r"""Structural vibration.

This example demonstrates the solution of a three-dimensional
vector-valued problem. For this purpose, we consider an elastic
eigenvalue problem.

The governing equation for the displacement of the elastic structure
:math:`\Omega` reads: find :math:`\boldsymbol{u} : \Omega \rightarrow
\mathbb{R}^3` satisfying

.. math::
   \rho \ddot{\boldsymbol{u}} = \mathrm{div}\,\boldsymbol{\sigma}(\boldsymbol{u}) + \rho \boldsymbol{g},
where :math:`\rho = 8050\,\frac{\mathrm{kg}}{\mathrm{m}^3}` is the
density, :math:`\boldsymbol{g}` is the gravitational acceleration and
:math:`\boldsymbol{\sigma}` is the linear elastic stress tensor
defined via

.. math::
   \begin{aligned}
   \boldsymbol{\sigma}(\boldsymbol{w}) &= 2 \mu \boldsymbol{\epsilon}(\boldsymbol{w}) + \lambda \mathrm{tr}\,\boldsymbol{\epsilon}(\boldsymbol{w}) \boldsymbol{I}, \\
   \boldsymbol{\epsilon}(\boldsymbol{w}) &= \frac12( \nabla \boldsymbol{w} + \nabla \boldsymbol{w}^T).
   \end{aligned}
Moreover, the Lamé parameters are given by

.. math::
   \lambda = \frac{E}{2(1 + \nu)}, \quad \mu = \frac{E \nu}{(1+ \nu)(1 - 2 \nu)},
where the Young's modulus :math:`E=200\cdot 10^9\,\text{Pa}`
and the Poisson ratio :math:`\nu = 0.3`.

We consider two kinds of boundary conditions. On a *fixed part* of the boundary, :math:`\Gamma_D \subset \partial \Omega`, the displacement field :math:`\boldsymbol{u}` satisfies

.. math::
   \boldsymbol{u}|_{\Gamma_D} = \boldsymbol{0}.
Moreover, on a *free part* of the boundary, :math:`\Gamma_N = \partial \Omega \setminus \Gamma_D`, the *traction vector* :math:`\boldsymbol{\sigma}(\boldsymbol{u})\boldsymbol{n}` satisfies

.. math::
   \boldsymbol{\sigma}(\boldsymbol{u})\boldsymbol{n} \cdot \boldsymbol{n}|_{\Gamma_N} = 0,
where :math:`\boldsymbol{n}` denotes the outward normal.

Neglecting the gravitational acceleration :math:`\boldsymbol{g}` and
assuming a periodic solution of the form

.. math::
   \boldsymbol{u}(\boldsymbol{x},t) = \boldsymbol{w}(\boldsymbol{x}) \sin \omega t,
leads to the following eigenvalue problem with :math:`\boldsymbol{w}` and :math:`\omega` as unknowns:

.. math::
   \mathrm{div}\,\boldsymbol{\sigma}(\boldsymbol{w}) = \rho \omega^2 \boldsymbol{w}.
The weak formulation of the problem reads: find :math:`(\boldsymbol{w},\omega) \in V \times \mathbb{R}` satisfying

.. math::
   (\boldsymbol{\sigma}(\boldsymbol{w}), \boldsymbol{\epsilon}(\boldsymbol{v})) = \rho \omega^2 (\boldsymbol{w}, \boldsymbol{v}) \quad \forall \boldsymbol{v} \in V,
where the variational space :math:`V` is defined as

.. math::
   V = \{ \boldsymbol{w} \in [H^1(\Omega)]^3 : \boldsymbol{w}|_{\Gamma_D} = \boldsymbol{0} \}.
The bilinear form for the problem can be found from
:func:`skfem.models.elasticity.linear_elasticity`.  Moreover, the mesh
for the problem is loaded from an external file *beams.msh*, which is
included in the source code distribution.

"""
from skfem import *
from skfem.models.elasticity import linear_elasticity,\
                                    lame_parameters
import numpy as np

from pathlib import Path

m = MeshTet.load(Path(__file__).parent / 'meshes' / 'beams.msh')
e1 = ElementTetP2()
e = ElementVectorH1(e1)

ib = CellBasis(m, e)

K = asm(linear_elasticity(*lame_parameters(200.0e9, 0.3)), ib)

rho = 8050.0


@BilinearForm
def mass(u, v, w):
    from skfem.helpers import dot
    return dot(rho * u, v)

M = asm(mass, ib)

L, x = solve(
    *condense(K, M, D=ib.find_dofs()["fixed"]), solver=solver_eigen_scipy_sym()
)

if __name__ == "__main__":
    from skfem.visuals.matplotlib import draw, show
    sf = 2.0
    draw(MeshTet(np.array(m.p + sf * x[ib.nodal_dofs, 0]), m.t))
    show()
