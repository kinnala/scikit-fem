from skfem import *
from skfem.models.elasticity import linear_elasticity
import numpy as np
from pygmsh import generate_mesh
from pygmsh.built_in import Geometry
from skfem.io import from_meshio
from skfem.io.json import from_file, to_file
import meshio


# create meshes
try:
    m = from_file("docs/examples/ex04_mesh.json")
except FileNotFoundError:
    geom = Geometry()
    points = []
    lines = []
    points.append(geom.add_point([0., 0., 0.], .1))
    points.append(geom.add_point([0., 1., 0.], .1))
    points.append(geom.add_point([0.,-1., 0.], .1))
    lines.append(geom.add_circle_arc(points[2], points[0], points[1]))
    geom.add_physical(lines[-1], 'contact')
    lines.append(geom.add_line(points[1], points[2]))
    geom.add_physical(lines[-1], 'dirichlet')
    geom.add_physical(geom.add_plane_surface(geom.add_line_loop(lines)), 'domain')
    m = from_meshio(generate_mesh(geom, dim=2))
    to_file(m, "docs/examples/ex04_mesh.json")

M = MeshLine(np.linspace(0, 1, 6)) * MeshLine(np.linspace(-1, 1, 10))
M.translate((1.0, 0.0))
M.refine()


# define elements and bases
e1 = ElementTriP2()
e = ElementVectorH1(e1)

E1 = ElementQuad2()
E = ElementVectorH1(E1)

ib = InteriorBasis(m, e, intorder=4)
Ib = InteriorBasis(M, E, intorder=4)
fb = FacetBasis(m, e)
Fb = FacetBasis(M, E)

mappings = MortarPair.init_2D(m, M,
                              m.boundaries['contact'],
                              M.facets_satisfying(lambda x: x[0] == 1.0),
                              np.array([0.0, 1.0]))

mb = [
    MortarBasis(m, e, mapping = mappings[0], intorder=4),
    MortarBasis(M, E, mapping = mappings[1], intorder=4)
]


# define bilinear forms
E1 = 1000.0
E2 = 1000.0

nu1 = 0.3
nu2 = 0.3

Mu1 = E1 / (2. * (1. + nu1))
Mu2 = E2 / (2. * (1. + nu2))

Lambda1 = E1 * nu1 / ((1. + nu1) * (1. - 2. * nu1))
Lambda2 = E2 * nu2 / ((1. + nu2) * (1. - 2. * nu2))

Mu = Mu1
Lambda = Lambda1

weakform1 = linear_elasticity(Lambda=Lambda, Mu=Mu)
weakform2 = linear_elasticity(Lambda=Lambda, Mu=Mu)


alpha = 1000
limit = 0.3


# assemble the stiffness matrices
K1 = asm(weakform1, ib)
K2 = asm(weakform2, Ib)
K = [[K1, 0.], [0., K2]]
f = [None] * 2

def tr(T):
    return T[0, 0] + T[1, 1]

def C(T):
    return np.array([[2. * Mu * T[0, 0] + Lambda * tr(T), 2. * Mu * T[0, 1]],
                     [2. * Mu * T[1, 0], 2. * Mu * T[1, 1] + Lambda * tr(T)]])

def Eps(dw):
    return np.array([[dw[0][0], .5 * (dw[0][1] + dw[1][0])],
                     [.5 * (dw[1][0] + dw[0][1]), dw[1][1]]])


# assemble the mortar matrices
for i in range(2):
    for j in range(2):
        @bilinear_form
        def bilin_penalty(u, du, v, dv, w):
            n = w.n
            ju = (-1.) ** i * (u[0] * n[0] + u[1] * n[1])
            jv = (-1.) ** j * (v[0] * n[0] + v[1] * n[1])
            mu = .5 * (n[0] * C(Eps(du))[0, 0] * n[0] +
                       n[0] * C(Eps(du))[0, 1] * n[1] +
                       n[1] * C(Eps(du))[1, 0] * n[0] +
                       n[1] * C(Eps(du))[1, 1] * n[1])
            mv = .5 * (n[0] * C(Eps(dv))[0, 0] * n[0] +
                       n[0] * C(Eps(dv))[0, 1] * n[1] +
                       n[1] * C(Eps(dv))[1, 0] * n[0] +
                       n[1] * C(Eps(dv))[1, 1] * n[1])
            h = w.h
            return (1. / (alpha * h) * ju * jv - mu * jv - mv * ju) *(np.abs(w.x[1]) <= limit)

        K[j][i] += asm(bilin_penalty, mb[i], mb[j])

    @linear_form
    def lin_penalty(v, dv, w):
        n = w.n
        jv = (-1.) ** i * (v[0] * n[0] + v[1] * n[1])
        mv = .5 * (n[0] * C(Eps(dv))[0, 0] * n[0] +
                   n[0] * C(Eps(dv))[0, 1] * n[1] +
                   n[1] * C(Eps(dv))[1, 0] * n[0] +
                   n[1] * C(Eps(dv))[1, 1] * n[1])
        h = w.h
        def gap(x):
            return (1. - np.sqrt(1. - x[1] ** 2))
        return (1. / (alpha * h) * gap(w.x) * jv - gap(w.x) * mv) * (np.abs(w.x[1]) <= limit)

    f[i] = asm(lin_penalty, mb[i])

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
        @bilinear_form
        def proj_cauchy(u, du, v, dv, w):
            return C(Eps(du))[itr, jtr] * v

        @bilinear_form
        def mass(u, du, v, dv, w):
            return u * v

        ib_dg = InteriorBasis(m, e_dg, intorder=4)
        Ib_dg = InteriorBasis(M, E_dg, intorder=4)

        s[itr, jtr] = solve(asm(mass, ib_dg), asm(proj_cauchy, ib, ib_dg) @ x[i1])
        S[itr, jtr] = solve(asm(mass, Ib_dg), asm(proj_cauchy, Ib, Ib_dg) @ x[i2])

s[2, 2] = nu1 * (s[0, 0] + s[1, 1])
S[2, 2] = nu2 * (S[0, 0] + S[1, 1])

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
    savefig(splitext(argv[0])[0] + '_vonmises.png')
