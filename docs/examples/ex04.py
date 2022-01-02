r"""Contact problem.

Mortar methods allow setting interface conditions on non-matching meshes.
They are useful also when solving variational inequalities such as
`elastic contact problems <https://arxiv.org/abs/1902.09312>`_.

This example solves the first contact iteration for the following prototype
contact problem: find :math:`\boldsymbol{u}_i : \Omega_i \rightarrow
\mathbb{R}^2`, :math:`i = 1,2`, such that

.. math::
   \begin{aligned}
        \boldsymbol{\mathrm{div}}\,\boldsymbol{\sigma}_i(\boldsymbol{u}_i)&=\boldsymbol{0} \quad && \text{in $\Omega_i$,} \\
        \boldsymbol{u}_1&=(0.1, 0) \quad && \text{on $\Gamma_{D,1}$,} \\
        \boldsymbol{u}_2&=\boldsymbol{0} \quad && \text{on $\Gamma_{D,2}$,} \\
        \boldsymbol{\sigma}_2(\boldsymbol{u}_2) \boldsymbol{n}_2 &=\boldsymbol{0} \quad && \text{on $\Gamma_{N,2}$,} \\
        \boldsymbol{\sigma}_{i,t}(\boldsymbol{u}_i) &= \boldsymbol{0} && \text{on $\Gamma$,} \\
       \sigma_{1,n}(\boldsymbol{u}_1(\boldsymbol{\gamma}(\boldsymbol{x})) - \sigma_{2,n}(\boldsymbol{u}_2)&=0 && \text{on $\Gamma$,} \\
        [[u_n]] - g   &\geq 0 && \text{on $\Gamma$,} \\
   \sigma_{2,n}(\boldsymbol{u}_2)&\leq 0 && \text{on $\Gamma$,} \\
          ([[u_n]] - g)  \sigma_{2,n}(\boldsymbol{u}_2) &= 0 && \text{on $\Gamma$,}
   \end{aligned}

where

* :math:`\Omega_1 = \{ (x, y) : x^2 + y^2 < 1 \} \setminus \{ (x, y) : x < 0\}`,
* :math:`\Omega_2 = (1, 2) \times (-1, 1)`,
* :math:`\Gamma_{D,1} = \{ (x, y) \in \Omega_1 : x=0 \}`,
* :math:`\Gamma_{D,2} = \{ (x, y) \in \Omega_2 : x=2 \}`,
* :math:`\Gamma = \{ (x, y) \in \Omega_2 : x=1 \}`,
* :math:`g((x,y)) = 1 - \sqrt{1 - y^2}`,
* :math:`\boldsymbol{\gamma} : \Gamma \rightarrow \{ (x, y) \in \partial
  \Omega_1 : x > 0 \}`, :math:`\boldsymbol{\gamma}((x,y)) = (g(x-1)+1, y)`,
* :math:`[[u_n]] = \boldsymbol{u}_1(\boldsymbol{\gamma}(\boldsymbol{x})) \cdot \boldsymbol{n} - \boldsymbol{u}_2(\boldsymbol{x}) \cdot \boldsymbol{n}`,

and the directions for evaluating :math:`[[u_n]]`, :math:`\sigma_{1,n}` and
:math:`\boldsymbol{\sigma}_{1,t}` are defined as :math:`\boldsymbol{n}=(1,0)`
and :math:`\boldsymbol{t}=(0,1)`.
This is a nonlinear problem since we do not know a priori which subset
:math:`\Gamma_C \subset \Gamma` satisfies :math:`([[u_n]] - g)|_{\Gamma_C} = 0`.

.. note::

   The example solves a simplified prototype problem.
   Instead of iterating for the true contact boundary,
   we solve a single contact iteration (a linear problem) with the initial
   guess :math:`\{ (x, y) \in \Gamma : |y| < 0.1 \}`.
   Solving a real contact problem involves repeatedly solving and guessing a new
   candidate boundary :math:`\Gamma_C` until convergence.
   Extending this example should be straightforward.

"""

from skfem import *
from skfem.models.elasticity import (linear_elasticity, lame_parameters,
                                     linear_stress)
from skfem.helpers import dot, ddot, prod, sym_grad, jump, mul
import numpy as np
from skfem.io import from_meshio
from skfem.io.json import from_file, to_file
from pathlib import Path

# create meshes
mesh_file = Path(__file__).parent / 'meshes' / 'ex04_mesh.json'
m = from_file(mesh_file)

M = (
    (MeshLine(np.linspace(1, 2, 6)) * MeshLine(np.linspace(-1, 1, 10)))
    .refined().with_boundaries({"contact": lambda x: x[0] == 1.0})
)

# define elements and bases
e1 = ElementTriP2()
e = ElementVector(e1)

E1 = ElementQuad2()
E = ElementVector(E1)

dofs1 = Dofs(m, e)
dofs2 = Dofs(M, E, offset=dofs1.N)

bases = [
    Basis(m, e, intorder=4, dofs=dofs1),
    Basis(M, E, intorder=4, dofs=dofs2),
]

mapping = MappingMortar.init_2D(m, M,
                                m.boundaries["contact"],
                                M.boundaries["contact"],
                                np.array([0.0, 1.0]))

mb = [
    MortarFacetBasis(m, e, mapping=mapping, intorder=4, side=0, dofs=dofs1),
    MortarFacetBasis(M, E, mapping=mapping, intorder=4, side=1, dofs=dofs2),
]

# define bilinear forms
youngs_modulus = 1000.0
poisson_ratio = 0.3
Lambda, Mu = lame_parameters(youngs_modulus, poisson_ratio)

weakform = linear_elasticity(Lambda, Mu)
C = linear_stress(Lambda, Mu)

alpha = 1000
limit = 0.3

# mortar forms
@BilinearForm
def bilin_mortar(u, v, w):
    # jumps
    ju, jv = jump(w, dot(u, w.n), dot(v, w.n))
    mu = .5 * dot(w.n, mul(C(sym_grad(u)), w.n))
    mv = .5 * dot(w.n, mul(C(sym_grad(v)), w.n))
    return ((1. / (alpha * w.h) * ju * jv - mu * jv - mv * ju)
            * (np.abs(w.x[1]) <= limit))

def gap(x):
    """Initial gap between the bodies."""
    return (1. - np.sqrt(1. - x[1] ** 2))


@LinearForm
def lin_mortar(v, w):
    jv = jump(w, dot(v, w.n))
    mv = .5 * ddot(prod(w.n, w.n), C(sym_grad(v)))
    return ((1. / (alpha * w.h) * gap(w.x) * jv - gap(w.x) * mv)
            * (np.abs(w.x[1]) <= limit))


# assemble
K = asm(weakform, bases) + asm(bilin_mortar, mb, mb)
f = asm(lin_mortar, mb)

# set boundary conditions and solve
i1 = np.arange(bases[0].N)
i2 = np.arange(bases[1].N) + bases[0].N

D1 = bases[0].get_dofs(lambda x: x[0] == 0.0).all()
D2 = bases[1].get_dofs(lambda x: x[0] == 2.0).all()

x = np.zeros(K.shape[0])
D = np.concatenate((D1, D2))
I = np.setdiff1d(np.arange(K.shape[0]), D)

x[bases[0].get_dofs(lambda x: x[0] == 0.0).nodal['u^1']] = 0.1
x[bases[0].get_dofs(lambda x: x[0] == 0.0).facet['u^1']] = 0.1

x = solve(*condense(K, f, I=I, x=x))

# create a displaced mesh
sf = 1

mdefo = m.translated(sf * x[dofs1.nodal_dofs])
Mdefo = M.translated(sf * x[dofs2.nodal_dofs])


# post processing
s, S = {}, {}
e_dg = ElementTriDG(ElementTriP1())
E_dg = ElementQuadDG(ElementQuad1())

for itr in range(2):
    for jtr in range(2):

        @BilinearForm
        def proj_cauchy(u, v, w):
            return C(sym_grad(u))[itr, jtr] * v

        @BilinearForm
        def mass(u, v, w):
            return u * v

        ib_dg = Basis(mdefo, e_dg, intorder=4)
        Ib_dg = Basis(Mdefo, E_dg, intorder=4)

        s[itr, jtr] = solve(asm(mass, ib_dg),
                            asm(proj_cauchy, bases[0], ib_dg) @ x[i1])
        S[itr, jtr] = solve(asm(mass, Ib_dg),
                            asm(proj_cauchy, bases[1], Ib_dg) @ x, I=i2)

s[2, 2] = poisson_ratio * (s[0, 0] + s[1, 1])
S[2, 2] = poisson_ratio * (S[0, 0] + S[1, 1])

vonmises1 = np.sqrt(.5 * ((s[0, 0] - s[1, 1]) ** 2 +
                          (s[1, 1] - s[2, 2]) ** 2 +
                          (s[2, 2] - s[0, 0]) ** 2 +
                          6. * s[0, 1] ** 2))

vonmises2 = np.sqrt(.5 * ((S[0, 0] - S[1, 1]) ** 2 +
                          (S[1, 1] - S[2, 2]) ** 2 +
                          (S[2, 2] - S[0, 0]) ** 2 +
                          6. * S[0, 1] ** 2))


def visualize():
    from skfem.visuals.matplotlib import plot, draw
    ax = plot(ib_dg,
              vonmises1,
              shading='gouraud',
              colorbar='$\sigma_{\mathrm{mises}}$')
    draw(mdefo, ax=ax)
    plot(Ib_dg, vonmises2, ax=ax, Nrefs=3, shading='gouraud')
    draw(Mdefo, ax=ax)
    return ax


if __name__ == "__main__":
    visualize().show()
