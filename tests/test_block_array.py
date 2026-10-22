import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfem import (BilinearForm, LinearForm, Functional, Basis, MeshTet,
                   MeshTri, ElementVector, ElementTetP2, ElementTriP2,
                   ElementTriP1, ElementComposite, asm)
from skfem.element import DiscreteField
from skfem.element.block_array import BlockArray, BlockDiscreteField
from skfem.helpers import (dot, ddot, grad, div, curl, sym_grad, trace,
                           transpose, eye, identity, mul, prod, inner, det,
                           inv)
from skfem.models.elasticity import linear_elasticity


def random_block(rng, shape, r, keys):
    return BlockArray(shape, r, {k: rng.random(shape[r:]) for k in keys})


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    a = random_block(rng, (3, 3, 4, 5), 1, [(1,)])
    b = random_block(rng, (3, 3, 4, 5), 2, [(0, 1), (2, 2), (1, 0)])
    c = rng.random((3, 3, 4, 5))
    return a, b, c


def test_dense_equivalence(arrays):
    a, b, c = arrays
    A, B = np.asarray(a), np.asarray(b)
    cases = [
        (ddot(a, b), ddot(A, B)),
        (ddot(a, c), ddot(A, c)),
        (ddot(c, b), ddot(c, B)),
        (transpose(a), transpose(A)),
        (sym_grad(DiscreteField(np.zeros(1), grad=c)),
         .5 * (c + transpose(c))),
        (.5 * (a + transpose(a)), .5 * (A + transpose(A))),
        (trace(b), trace(B)),
        (mul(a, b[0]), mul(A, B[0])),
        (mul(c, a[1]), mul(c, A[1])),
        (prod(a[1], b[2]), prod(A[1], B[2])),
        (np.einsum('ij...,jk...->ik...', a, b),
         np.einsum('ij...,jk...->ik...', A, B)),
        (a + b, A + B),
        (a - b, A - B),
        (b - c, B - c),
        (a * b, A * B),
        (a * c, A * c),
        (c[0, 0] * a, c[0, 0] * A),
        (a / c[0, 0], A / c[0, 0]),
        (2. * a - b / 3., 2. * A - B / 3.),
        (-a, -A),
        (a[1], A[1]),
        (a[0], A[0]),
        (a[1, 2], A[1, 2]),
        (a[0, 2], A[0, 2]),
        (b[2, 2, 1], B[2, 2, 1]),
        (a[:, 1], A[:, 1]),
        (eye(c[0, 0], 3) + b, np.eye(3)[:, :, None, None] * c[0, 0] + B),
        (identity(c[0, 0], 3), np.eye(3)[:, :, None, None]
         * np.ones((4, 5))),
        (np.sin(a), np.sin(A)),
        (np.cos(a), np.cos(A)),
        (det(b), det(B)),
        (inv(b + identity(c[0, 0], 3)), inv(B + identity(c[0, 0], 3))),
        (np.sum(a, axis=0), np.sum(A, axis=0)),
        (a.T, A.T),
        (sum(x for x in a), sum(x for x in A)),
    ]
    for new, ref in cases:
        assert np.asarray(new).shape == ref.shape
        assert_allclose(np.asarray(new), ref)


def test_inplace(arrays):
    a, b, _ = arrays
    out = 0. * a
    out += b
    out *= 2.
    assert_allclose(np.asarray(out), 2. * np.asarray(b))


def test_zeros():
    z = BlockArray.zeros((3, 4, 5))
    assert isinstance(z * np.ones((4, 5)), BlockArray)
    assert len((z * np.ones((4, 5))).blocks) == 0
    assert_allclose(z + np.ones((3, 4, 5)), np.ones((3, 4, 5)))
    assert_allclose(z - np.ones((3, 4, 5)), -np.ones((3, 4, 5)))
    assert_allclose(np.ones((3, 4, 5)) - z, np.ones((3, 4, 5)))
    assert_allclose(np.asarray(np.cos(z)), np.ones((3, 4, 5)))
    assert len(dot(z, np.ones((3, 4, 5))).blocks) == 0


def test_index_error():
    a = BlockArray((2, 4, 5), 1, {(0,): np.ones((4, 5))})
    with pytest.raises(IndexError):
        a[2]
    assert len(list(a)) == 2


def densify(f):
    if isinstance(f, tuple):
        return tuple(densify(c) for c in f)
    if isinstance(f, BlockDiscreteField):
        return DiscreteField(*(None if c is None else np.asarray(c)
                               for c in f.astuple))
    return f


def dense_bilinear(form):
    return BilinearForm(lambda u, v, w: form.form(densify(u),
                                                  densify(v),
                                                  w))


@BilinearForm
def mass_manual(u, v, w):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@BilinearForm
def vector_laplace(u, v, w):
    return (ddot(grad(u), grad(v)) + div(u) * div(v)
            + dot(curl(u), curl(v)) + inner(u, v))


@pytest.mark.parametrize('form', [
    linear_elasticity(1., 2.),
    mass_manual,
    vector_laplace,
])
def test_vector_assembly(form):
    basis = Basis(MeshTet(), ElementVector(ElementTetP2()))
    assert isinstance(basis.basis[0][0], BlockDiscreteField)
    A = form.assemble(basis)
    B = dense_bilinear(form).assemble(basis)
    assert abs(A - B).max() < 1e-12 * abs(B).max()


def test_vector_linear_and_functional():
    basis = Basis(MeshTet(), ElementVector(ElementTetP2()))

    @LinearForm
    def load(v, w):
        return v[2] * w.x[0] + dot(v, w.x)

    x = load.assemble(basis)
    y = LinearForm(lambda v, w: load.form(densify(v), w)).assemble(basis)
    assert_allclose(x, y)

    @Functional
    def energy(w):
        return ddot(grad(w['u']), grad(w['u']))

    u = basis.interpolate(np.arange(basis.N, dtype=np.float64))
    assert isinstance(u, DiscreteField)
    assert_allclose(energy.assemble(basis, u=u),
                    (np.einsum('ij...,ij...', u.grad, u.grad)
                     * basis.dx).sum())


def test_tensor_assembly():
    basis = Basis(MeshTri(),
                  ElementVector(ElementVector(ElementTriP2())))

    @BilinearForm
    def form(u, v, w):
        return ddot(u, v) + ddot(transpose(u), v) + trace(u) * trace(v)

    assert basis.basis[0][0].r == 2
    A = form.assemble(basis)
    B = dense_bilinear(form).assemble(basis)
    assert abs(A - B).max() < 1e-12 * abs(B).max()


def test_composite_assembly():
    basis = Basis(MeshTri().refined(),
                  ElementComposite(ElementVector(ElementTriP2()),
                                   ElementTriP1()))

    @BilinearForm
    def stokes(u, p, v, q, w):
        return ddot(grad(u), grad(v)) - div(u) * q - div(v) * p - p * q

    A = stokes.assemble(basis)
    B = BilinearForm(lambda u, p, v, q, w: stokes.form(
        *densify((u, p, v, q)), w)).assemble(basis)
    assert abs(A - B).max() < 1e-12 * abs(B).max()

    # zero components are not stored
    u_field = basis.basis[2][0]  # third DOF belongs to p
    assert isinstance(u_field, BlockDiscreteField)
    assert len(u_field.blocks) == 0
    assert len(u_field.grad.blocks) == 0

    # block of a form
    C1 = asm(stokes.block(1, 1), Basis(basis.mesh, ElementTriP1()))
    C2 = A[basis.nodal_dofs[-1]].T[basis.nodal_dofs[-1]].T
    assert abs(C1 - C2).max() < 1e-12
