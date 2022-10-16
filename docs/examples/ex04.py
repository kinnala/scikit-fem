r"""Contact problem.

Mortar methods allow setting interface conditions on non-matching meshes.
They are useful also when solving variational inequalities such as
`elastic contact problems <https://arxiv.org/abs/1902.09312>`_.

"""

import numpy as np
from skfem import *
from skfem.supermeshing import intersect
from skfem.models.elasticity import (linear_elasticity, lame_parameters,
                                     linear_stress)
from skfem.helpers import dot, ddot, prod, sym_grad, jump, mul
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

# create trace meshes and project
p1, t1, facets1 = m1.trace('contact')
p2, t2, facets2 = m2.trace('contact')
p1 = p1[1:]
p2 = p2[1:]

# create supermesh for integration
p12, t12, orig1, orig2 = intersect(p1, t1, p2, t2)
m12 = MeshLine(p12, t12)

# create mappings
map1t = MeshLine(p1, t1).mapping()
map2t = MeshLine(p2, t2).mapping()
map1 = m1.mapping()
map2 = m2.mapping()
map12 = m12.mapping()

basis1 = Basis(m1, e1)
basis2 = Basis(m2, e2)

# quadrature points are created using the supermesh and mapped to the facets
# of the original meshes
X, W = get_quadrature(ElementLineP1, 3)

fbasis1 = FacetBasis(
    m1,
    e1,
    mapping=map1,
    quadrature=(
        map1t.invF(map12.F(X), tind=orig1),
        map12.detDF(X) / map1.detDG(X, find=facets1[orig1]) * W,
    ),
    facets=facets1[orig1],
)

fbasis2 = FacetBasis(
    m2,
    e2,
    mapping=map2,
    quadrature=(
        map2t.invF(map12.F(X), tind=orig2),
        map12.detDF(X) / map2.detDG(X, find=facets2[orig2]) * W,
    ),
    facets=facets2[orig2],
)

# problem definition
youngs_modulus = 1000.0
poisson_ratio = 0.3

weakform = linear_elasticity(*lame_parameters(youngs_modulus, poisson_ratio))
C = linear_stress(*lame_parameters(youngs_modulus, poisson_ratio))

alpha = 1000
limit = 0.3

# mortar forms
@BilinearForm
def bilin_mortar(u, v, w):
    ju, jv = jump(w, dot(u, w.n), dot(v, w.n))
    mu = .5 * dot(w.n, mul(C(sym_grad(u)), w.n))
    mv = .5 * dot(w.n, mul(C(sym_grad(v)), w.n))
    return ((1. / (alpha * w.h) * ju * jv - mu * jv - mv * ju)
            * (np.abs(w.x[1]) <= limit))

def gap(x):
    return (1. - np.sqrt(1. - x[1] ** 2))

@LinearForm
def lin_mortar(v, w):
    jv = jump(w, dot(v, w.n))
    mv = .5 * dot(w.n, mul(C(sym_grad(v)), w.n))
    return ((1. / (alpha * w.h) * gap(w.x) * jv - gap(w.x) * mv)
            * (np.abs(w.x[1]) <= limit))

params = {
    'h': map12.detDF(X),
    'n': -fbasis2.normals,
}

# assemble the block system
K1 = weakform.assemble(basis1)
K2 = weakform.assemble(basis2)

K = [[K1, 0], [0, K2]]
f = [None, None]

fbases = [fbasis1, fbasis2]
for i in [0, 1]:
    for j in [0, 1]:
        K[i][j] += bilin_mortar.assemble(fbases[j],
                                         fbases[i],
                                         idx=(j, i),
                                         **params)
    f[i] = lin_mortar.assemble(fbases[i], idx=(i,), **params)

K = bmat(K, 'csr')
f = np.concatenate(f)

D1 = basis1.get_dofs('dirichlet').all()
D2 = basis2.get_dofs('dirichlet').all() + K.blocks[0]

# initialize boundary conditions
y1 = basis1.zeros()
y2 = basis2.zeros()
y1[basis1.get_dofs('dirichlet').all('u^1')] = .1
y = np.concatenate((y1, y2))

# linear solve
y = solve(*condense(K, f, D=np.concatenate((D1, D2)), x=y))

# create a displaced mesh for visualization
sf = 1
y1, y2 = np.split(y, K.blocks)
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
