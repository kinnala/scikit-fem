import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfem import (MeshTri, MeshTri2, MeshTet, MeshLine1, MeshQuad,
                   ElementTriP1, ElementTriP2, ElementQuad1, ElementLineP1,
                   ElementVector, Basis, FacetBasis, BilinearForm, LinearForm,
                   Functional, solve, condense)
from skfem.mapping import MappingManifold
from skfem.helpers import dot, grad


@Functional
def area(w):
    return 1. + 0. * w.x[0]


def init_sphere(nrefs):
    m = MeshTet.init_tensor(*(3 * [np.linspace(-1, 1, 3)]))
    f = m.facets[:, m.boundary_facets()]
    ix, t = np.unique(f, return_inverse=True)
    m = MeshTri(m.p[:, ix], t.reshape(f.shape))
    for _ in range(nrefs):
        m = m.refined()
        m = MeshTri(m.p / np.linalg.norm(m.p, axis=0), m.t)
    return m


def test_default_mapping():
    m = MeshTri(np.array([[0., 1., 0.],
                          [0., 0., 1.],
                          [0., 0., 0.]]), np.array([[0], [1], [2]]))
    assert isinstance(m.mapping(), MappingManifold)
    assert not isinstance(MeshTri().mapping(), MappingManifold)


@pytest.mark.parametrize(
    "m,e",
    [
        (MeshTri.init_sqsymmetric().refined(), ElementTriP1()),
        (MeshTri.init_sqsymmetric().refined(), ElementTriP2()),
        (MeshQuad.init_tensor(np.linspace(0, 1, 4),
                              np.linspace(0, 2, 3)), ElementQuad1()),
    ]
)
def test_rotated_plane(m, e):
    """A rotated planar 2D mesh in 3D reproduces the 2D results."""
    rot = np.array([[1., 0.], [0., np.cos(.3)], [0., np.sin(.3)]])
    rot = np.array([[np.cos(.5), -np.sin(.5), 0],
                    [np.sin(.5), np.cos(.5), 0],
                    [0, 0, 1]]) @ rot
    m3 = type(m)(rot @ m.p, m.t)
    b2 = Basis(m, e)
    b3 = Basis(m3, e)

    laplace = BilinearForm(lambda u, v, w: dot(grad(u), grad(v)) + u * v)
    assert_allclose(laplace.assemble(b3).toarray(),
                    laplace.assemble(b2).toarray(), atol=1e-12)
    assert_allclose(area.assemble(b3), area.assemble(b2))

    # gradients are tangential
    n = b3.default_parameters()['n']
    assert_allclose(np.einsum('i...,i...', b3.basis[1][0].grad, n), 0.,
                    atol=1e-12)

    # boundary integrals
    fb2 = FacetBasis(m, e)
    fb3 = FacetBasis(m3, e)
    assert_allclose(area.assemble(fb3), area.assemble(fb2))
    # conormals are rotated normals
    assert_allclose(np.asarray(fb3.normals),
                    np.einsum('ij,j...->i...', rot, np.asarray(fb2.normals)),
                    atol=1e-12)


def test_laplace_beltrami():
    """-Δu + u = 3x on unit sphere has the solution u = x."""
    errs = []
    for nrefs in [2, 3]:
        m = MeshTri2.from_mesh(init_sphere(nrefs))
        m = MeshTri2(m.doflocs / np.linalg.norm(m.doflocs, axis=0), m.t)
        b = Basis(m, ElementTriP2())
        A = BilinearForm(
            lambda u, v, w: dot(grad(u), grad(v)) + u * v
        ).assemble(b)
        f = LinearForm(lambda v, w: 3. * w.x[0] * v).assemble(b)
        u = solve(A, f)
        errs.append(np.sqrt(Functional(
            lambda w: (w.u - w.x[0]) ** 2
        ).assemble(b, u=b.interpolate(u))))
        assert_allclose(area.assemble(b), 4. * np.pi, rtol=1e-4)
    assert errs[1] < 1e-5
    assert np.log2(errs[0] / errs[1]) > 2.9


def test_sphere_normals():
    b = Basis(init_sphere(2), ElementTriP1())
    n = np.asarray(b.default_parameters()['n'])
    assert_allclose(np.linalg.norm(n, axis=0), 1.)
    x = np.asarray(b.default_parameters()['x'])
    # normals are parallel to the position vector up to orientation
    assert (np.abs(np.sum(n * x, axis=0)
                   / np.linalg.norm(x, axis=0)) > .98).all()


def test_planar_truss():
    """Two-bar truss with an analytical solution."""
    m = MeshLine1(np.array([[0., 1., 2.],
                            [0., 1., 0.]]),
                  np.array([[0, 1], [1, 2]]))
    b = Basis(m, ElementVector(ElementLineP1(), 2))

    @BilinearForm
    def truss(u, v, w):
        return dot(dot(grad(u), w.t), w.t) * dot(dot(grad(v), w.t), w.t)

    K = truss.assemble(b)
    f = np.zeros(b.N)
    apex = b.get_dofs(nodes=np.array([1]))
    f[apex.nodal['u^2']] = -1.
    u = solve(*condense(K, f, D=b.get_dofs(nodes=np.array([0, 2]))))
    assert_allclose(u[apex.flatten()], [0., -np.sqrt(2.)], atol=1e-12)


def test_space_curve():
    s = np.linspace(0, 1, 5)
    m = MeshLine1(np.array([s, 2 * s, 2 * s]),
                  np.array([np.arange(4), np.arange(1, 5)]))
    b = Basis(m, ElementLineP1())
    assert_allclose(area.assemble(b), 3.)
    assert_allclose(np.asarray(b.default_parameters()['t'])[:, 0, 0],
                    [1. / 3, 2. / 3, 2. / 3])
    fb = FacetBasis(m, ElementLineP1())
    assert_allclose(area.assemble(fb), 2.)
    assert_allclose(np.sort(np.asarray(fb.normals)[0, :, 0]),
                    [-1. / 3, 1. / 3])
