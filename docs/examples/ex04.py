r"""Contact problem.

Mortar methods allow setting interface conditions on non-matching meshes.
They are useful also when solving variational inequalities such as
`elastic contact problems <https://arxiv.org/abs/1902.09312>`_.

"""

import numpy as np
from skfem import *
from skfem.supermeshing import mortar_quadrature
from skfem.models.elasticity import (linear_elasticity, lame_parameters,
                                     linear_stress)
from skfem.helpers import dot, sym_grad, jump, mul
from skfem.io.json import from_file
from pathlib import Path


# create meshes
mesh_file = Path(__file__).parent / 'meshes' / 'ex04_mesh.json'
m1 = from_file(mesh_file)
m2 = (
    (MeshLine(np.linspace(1, 2, 6)) * MeshLine(np.linspace(-1, 1, 10)))
    .refined()
    .with_boundaries({
        'contact': lambda x: x[0] == 1.0,
        'dirichlet': lambda x: x[0] == 2.0,
    })
)

e1 = ElementVector(ElementTriP2())
e2 = ElementVector(ElementQuad2())

basis1, basis2 = Basis(m1, e1), Basis(m2, e2)

# mortaring
qw1, find1, qw2, find2 = mortar_quadrature(m1, 'contact',
                                           m2, 'contact', 0.1, slave=2)

fbasis1 = FacetBasis(m1, e1, quadrature=qw1, facets=find1)
fbasis2 = FacetBasis(m2, e2, quadrature=qw2, facets=find2)
fbasis = fbasis1 * fbasis2

# problem definition
youngs_modulus = 1000.0
poisson_ratio = 0.3

weakform = linear_elasticity(*lame_parameters(youngs_modulus, poisson_ratio))
C = linear_stress(*lame_parameters(youngs_modulus, poisson_ratio))

alpha = 1000
limit = 0.3

# mortar forms
@BilinearForm
def bilin_mortar(u1, u2, v1, v2, w):
    ju = dot(u1 - u2, w.n)
    jv = dot(v1 - v2, w.n)
    mu = .5 * (dot(w.n, mul(C(sym_grad(u1)), w.n))
               + dot(w.n, mul(C(sym_grad(u2)), w.n)))
    mv = .5 * (dot(w.n, mul(C(sym_grad(v1)), w.n))
               + dot(w.n, mul(C(sym_grad(v2)), w.n)))
    return ((1. / (alpha * w.h) * ju * jv - mu * jv - mv * ju)
            * (np.abs(w.x[1]) <= limit))

def gap(x):
    return (1. - np.sqrt(1. - x[1] ** 2))

@LinearForm
def lin_mortar(v1, v2, w):
    jv = dot(v1 - v2, w.n)
    mv = .5 * (dot(w.n, mul(C(sym_grad(v1)), w.n))
               + dot(w.n, mul(C(sym_grad(v2)), w.n)))
    return ((1. / (alpha * w.h) * gap(w.x) * jv - gap(w.x) * mv)
            * (np.abs(w.x[1]) <= limit))

# fix mesh parameter and normals from m2
params = {
    'h': fbasis2.mesh_parameters(),
    'n': -fbasis2.normals,
}

# assemble the block system
A1 = asm(weakform, basis1)
A2 = asm(weakform, basis2)
B = asm(bilin_mortar, fbasis, **params)
f = asm(lin_mortar, fbasis, **params)

K = bmat([[A1, None],
          [None, A2]], 'csr') + B

D1 = basis1.get_dofs('dirichlet').all()
D2 = basis2.get_dofs('dirichlet').all() + basis1.N

# initialize boundary conditions
y1 = basis1.zeros()
y2 = basis2.zeros()
y1[basis1.get_dofs('dirichlet').all('u^1')] = .1
y = np.concatenate((y1, y2))

# linear solve
y = solve(*condense(K, f, D=np.concatenate((D1, D2)), x=y))

# create a displaced mesh for visualization
sf = 1
(y1, _), (y2, _) = fbasis.split(y)
mdefo1 = m1.translated(sf * y1[basis1.nodal_dofs])
mdefo2 = m2.translated(sf * y2[basis2.nodal_dofs])

# calculate von Mises stress
s1, s2 = {}, {}
dg1 = basis1.with_element(ElementTriDG(ElementTriP1()))
dg2 = basis2.with_element(ElementQuadDG(ElementQuad1()))
u1 = basis1.interpolate(y1)
u2 = basis2.interpolate(y2)

for i in [0, 1]:
    for j in [0, 1]:
        s1[i, j] = dg1.project(C(sym_grad(u1))[i, j])
        s2[i, j] = dg2.project(C(sym_grad(u2))[i, j])

s1[2, 2] = poisson_ratio * (s1[0, 0] + s1[1, 1])
s2[2, 2] = poisson_ratio * (s2[0, 0] + s2[1, 1])

vonmises1 = np.sqrt(.5 * ((s1[0, 0] - s1[1, 1]) ** 2 +
                          (s1[1, 1] - s1[2, 2]) ** 2 +
                          (s1[2, 2] - s1[0, 0]) ** 2 +
                          6. * s1[0, 1] ** 2))

vonmises2 = np.sqrt(.5 * ((s2[0, 0] - s2[1, 1]) ** 2 +
                          (s2[1, 1] - s2[2, 2]) ** 2 +
                          (s2[2, 2] - s2[0, 0]) ** 2 +
                          6. * s2[0, 1] ** 2))


def visualize():
    from skfem.visuals.matplotlib import plot, draw
    pdg1 = Basis(mdefo1, dg1.elem)
    pdg2 = Basis(mdefo2, dg2.elem)
    ax = plot(pdg1,
              vonmises1,
              shading='gouraud',
              colorbar=r"$\sigma_{\mathrm{mises}}$")
    draw(mdefo1, ax=ax)
    plot(pdg2, vonmises2, ax=ax, nrefs=3, shading='gouraud')
    draw(mdefo2, ax=ax)
    return ax


if __name__ == "__main__":
    visualize().show()
