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
from skfem.models.elasticity import linear_elasticity,\
    lame_parameters, linear_stress
from skfem.models.helpers import dot, ddot,\
    prod, sym_grad
import numpy as np
from skfem.io import from_meshio
from skfem.io.json import from_file, to_file
from pathlib import Path

# create meshes
mesh_file = Path(__file__).parent / 'meshes' / 'ex04_mesh.json'
m = from_file(mesh_file)

M = (
    (MeshLine(np.linspace(0, 1, 6)) * MeshLine(np.linspace(-1, 1, 10)))
    .translated((1.0, 0.0))
    .refined()
)

# define elements and bases
e1 = ElementTriP2()
e = ElementVectorH1(e1)

E1 = ElementQuad2()
E = ElementVectorH1(E1)

ib = InteriorBasis(m, e, intorder=4)
Ib = InteriorBasis(M, E, intorder=4)

mapping = MappingMortar.init_2D(m, M,
                                m.boundaries['contact'],
                                M.facets_satisfying(lambda x: x[0] == 1.0),
                                np.array([0.0, 1.0]))

mb = [
    FacetBasis(m, e, mapping=mapping, intorder=4, side=0),
    FacetBasis(M, E, mapping=mapping, intorder=4, side=1),
]

# define bilinear forms
E = 1000.0
nu = 0.3
Lambda, Mu = lame_parameters(E, nu)

weakform1 = linear_elasticity(Lambda, Mu)
weakform2 = linear_elasticity(Lambda, Mu)
C = linear_stress(Lambda, Mu)

alpha = 1000
limit = 0.3

# assemble the stiffness matrices
K1 = asm(weakform1, ib)
K2 = asm(weakform2, Ib)
K = [[K1, 0.], [0., K2]]
f = [None] * 2


def gap(x):
    """Initial gap between the bodies."""
    return (1. - np.sqrt(1. - x[1] ** 2))

# assemble the mortar matrices
for i in range(2):
    for j in range(2):

        @BilinearForm
        def bilin_mortar(u, v, w):
            ju = (-1.) ** j * dot(u, w.n)
            jv = (-1.) ** i * dot(v, w.n)
            nxn = prod(w.n, w.n)
            mu = .5 * ddot(nxn, C(sym_grad(u)))
            mv = .5 * ddot(nxn, C(sym_grad(v)))
            return ((1. / (alpha * w.h) * ju * jv - mu * jv - mv * ju)
                    * (np.abs(w.x[1]) <= limit))

        K[i][j] += asm(bilin_mortar, mb[j], mb[i])

    @LinearForm
    def lin_mortar(v, w):
        jv = (-1.) ** i * dot(v, w.n)
        mv = .5 * ddot(prod(w.n, w.n), C(sym_grad(v)))
        return ((1. / (alpha * w.h) * gap(w.x) * jv - gap(w.x) * mv)
                * (np.abs(w.x[1]) <= limit))

    f[i] = asm(lin_mortar, mb[i])

import scipy.sparse
K = (scipy.sparse.bmat(K)).tocsr()

# set boundary conditions and solve
i1 = np.arange(K1.shape[0])
i2 = np.arange(K2.shape[0]) + K1.shape[0]

D1 = ib.get_dofs(lambda x: x[0] == 0.0).all()
D2 = Ib.get_dofs(lambda x: x[0] == 2.0).all()

x = np.zeros(K.shape[0])

f = np.hstack((f[0], f[1]))

x = np.zeros(K.shape[0])
D = np.concatenate((D1, D2 + ib.N))
I = np.setdiff1d(np.arange(K.shape[0]), D)

x[ib.get_dofs(lambda x: x[0] == 0.0).nodal['u^1']] = 0.1
x[ib.get_dofs(lambda x: x[0] == 0.0).facet['u^1']] = 0.1

x = solve(*condense(K, f, I=I, x=x))


# create a displaced mesh
sf = 1

m.p[0, :] = m.p[0, :] + sf * x[i1][ib.nodal_dofs[0, :]]
m.p[1, :] = m.p[1, :] + sf * x[i1][ib.nodal_dofs[1, :]]

M.p[0, :] = M.p[0, :] + sf * x[i2][Ib.nodal_dofs[0, :]]
M.p[1, :] = M.p[1, :] + sf * x[i2][Ib.nodal_dofs[1, :]]


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

        ib_dg = InteriorBasis(m, e_dg, intorder=4)
        Ib_dg = InteriorBasis(M, E_dg, intorder=4)

        s[itr, jtr] = solve(asm(mass, ib_dg),
                            asm(proj_cauchy, ib, ib_dg) @ x[i1])
        S[itr, jtr] = solve(asm(mass, Ib_dg),
                            asm(proj_cauchy, Ib, Ib_dg) @ x[i2])

s[2, 2] = nu * (s[0, 0] + s[1, 1])
S[2, 2] = nu * (S[0, 0] + S[1, 1])

vonmises1 = np.sqrt(.5 * ((s[0, 0] - s[1, 1]) ** 2 +
                          (s[1, 1] - s[2, 2]) ** 2 +
                          (s[2, 2] - s[0, 0]) ** 2 +
                          6. * s[0, 1]**2))

vonmises2 = np.sqrt(.5 * ((S[0, 0] - S[1, 1]) ** 2 +
                          (S[1, 1] - S[2, 2]) ** 2 +
                          (S[2, 2] - S[0, 0]) ** 2 +
                          6. * S[0, 1]**2))


if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import *

    ax = plot(ib_dg, vonmises1, shading='gouraud')
    draw(m, ax=ax)
    plot(Ib_dg, vonmises2, ax=ax, Nrefs=3, shading='gouraud')
    draw(M, ax=ax)
    savefig(splitext(argv[0])[0] + '_solution.png')
