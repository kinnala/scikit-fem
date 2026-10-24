import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal,
                           assert_allclose)

from skfem import (Basis, FacetBasis, MeshTri, MeshQuad, MeshTet, MeshLine,
                   ElementTriP1, ElementTriP2, ElementQuad1, ElementTetP1,
                   ElementTetP2, ElementVector, solve, condense, enforce,
                   bmat)
from skfem.autodiff import NonlinearForm
import skfem.autodiff.helpers as ngh
from skfem.autodiff.helpers import (grad, dot, ddot, mul, div, sym_grad,
                                    transpose, eye, trace, det, inv,
                                    cross, prod, identity, inner, stack)


def test_linear_poisson():

    m = MeshTri().refined(4)
    basis = Basis(m, ElementTriP1())
    x = basis.zeros()
    xe = basis.project(lambda x: (1 / (2. * np.pi ** 2)
                                  * (np.sin(np.pi * x[0])
                                     * np.sin(np.pi * x[1]))))

    @NonlinearForm
    def poisson(u, v, w):
        f = np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1])
        return dot(grad(u), grad(v)) - f * v

    for itr in range(10):
        xp = x.copy()
        x += solve(*condense(*poisson.assemble(basis, x=x),
                             D=basis.get_dofs()))
        if np.linalg.norm(x - xp) < 1e-8:
            break

    assert_array_almost_equal(x, xe, decimal=3)


def test_obstacle_problem():

    m = MeshTri().refined(3)
    basis = Basis(m, ElementTriP1())
    x = basis.zeros()

    @NonlinearForm
    def poisson(u, v, w):
        f = np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1])
        return (dot(grad(u), grad(v))
                + 1e5 * np.maximum(u - 0.01, 0) * v
                - f * v)

    for itr in range(10):
        xp = x.copy()
        x += solve(*condense(*poisson.assemble(basis, x=x),
                             D=basis.get_dofs()))
        if np.linalg.norm(x - xp) < 1e-8:
            break

    assert np.sum(x[x > 0.01] - 0.01) < 3e-4


def test_navier_stokes():

    m = MeshTri.init_sqsymmetric().refined(2).with_defaults()
    basis = Basis(m, ElementVector(ElementTriP2()) * ElementTriP1())
    x = basis.zeros()

    @NonlinearForm
    def navierstokes(u, p, v, q, w):
        return (ddot(sym_grad(u), sym_grad(v))
                + dot(mul(grad(u), u), v)
                - div(u) * q - div(v) * p - 1e-3 * p * q)

    x[basis.get_dofs('top').all('u^1^1')] = 100.

    for itr in range(50):
        xp = x.copy()
        x += solve(*condense(*navierstokes.assemble(basis, x=x),
                             D=basis.get_dofs().all(['u^1^1', 'u^2^1'])))
        if np.linalg.norm(x - xp) < 1e-8:
            break

    (u, ubasis), (p, pbasis) = basis.split(x)

    assert_almost_equal(np.max(p), 5212.45466, decimal=5)


def test_nonlin_elast():

    m = (MeshQuad
         .init_tensor(np.linspace(0, 5, 20),
                      np.linspace(0, 0.5, 5))
         .to_meshtri(style='x')
         .with_defaults())
    e = ElementVector(ElementTriP1())
    basis = Basis(m, e)
    x = basis.zeros()

    @NonlinearForm
    def elast(u, v, w):
        epsu = .5 * (grad(u) + transpose(grad(u))
                     + mul(transpose(grad(u)), grad(u)))
        epsv = .5 * (grad(v) + transpose(grad(v)))
        sigu = 2 * 10 * epsu + 1. * eye(trace(epsu), 2)
        return ddot(sigu, epsv) - w.t * 2e-2 * v[1]

    for itr in range(50):
        xp = x.copy()
        x += solve(*condense(*elast.assemble(basis,
                                             x=x,
                                             t=np.minimum((itr + 1) / 5, 1)),
                             D=basis.get_dofs({'left'}).all()))
        if np.linalg.norm(x - xp) < 1e-8:
            break

    assert_almost_equal(np.max(x), 2.83411524813795)


def _scalar_form(u, v, w):
    U = u * 1.
    return (np.exp(U) * dot((1. + U ** 2) * grad(u), grad(v))
            + np.sin(U) / (2. + U) * v
            + np.sqrt(1. + U * U) * np.cos(w.x[0]) * v
            + np.arctan(U) ** 3 * v
            + 2. ** U * np.log1p(U ** 2) * v)


def _vector_form(u, v, w):
    F = grad(u) + identity(grad(u))
    C = mul(transpose(F), F)
    S = (inv(C) * det(F) ** 2
         + np.tanh(trace(C)) * eye(np.ones_like(u[0]), 3)
         + prod(u, u) / (1. + inner(u, u)))
    return (ddot(mul(F, S), grad(v))
            + dot(cross(u, grad(u)[0]), v)
            + np.maximum(u[2], 0.) ** 2 * v[2])


def _vector_energy(u, w):
    F = grad(u) + identity(grad(u))
    J = det(F)
    return (.5 * (trace(mul(transpose(F), F)) - 3.)
            - np.log(J) + .5 * np.log(J) ** 2
            + 1e-2 * dot(u, u) ** 1.5 + np.abs(u[0] - 1.) * u[1])


def _fd_check(form, basis, x, hessian=False):
    form = NonlinearForm(hessian=hessian)(form)
    jac, rhs = form.assemble(basis, x=x)
    dx = np.random.RandomState(1).rand(basis.N)
    eps = 1e-6
    _, rhsp = form.assemble(basis, x=x + eps * dx)
    _, rhsm = form.assemble(basis, x=x - eps * dx)
    # rhs is the negative residual
    assert_allclose(jac @ dx, -(rhsp - rhsm) / (2 * eps),
                    rtol=1e-6, atol=1e-6 * np.abs(jac @ dx).max())
    return jac, rhs


@pytest.mark.parametrize(
    "form,basis,hessian",
    [
        (_scalar_form, Basis(MeshTri().refined(2), ElementTriP2()), False),
        (_vector_form,
         Basis(MeshTet().refined(1), ElementVector(ElementTetP1())),
         False),
        (_vector_energy,
         Basis(MeshTet().refined(1), ElementVector(ElementTetP1())),
         True),
    ]
)
def test_finite_difference(form, basis, hessian):
    x = .05 * np.random.RandomState(0).rand(basis.N) - .02
    jac, _ = _fd_check(form, basis, x, hessian=hessian)
    if hessian:
        # second variation of an energy is symmetric
        assert_allclose(jac.toarray(), jac.T.toarray(), atol=1e-12)


def test_mixed_finite_difference():
    basis = Basis(MeshTri().refined(2),
                  ElementVector(ElementTriP2()) * ElementTriP1())

    def form(u, p, v, q, w):
        return (ddot(sym_grad(u), sym_grad(v))
                + dot(mul(grad(u), u), v)
                + np.exp(p) * dot(u, grad(q))
                - div(u) * q - div(v) * p ** 3)

    x = .1 * np.random.RandomState(0).rand(basis.N)
    _fd_check(form, basis, x)


def _sparse_dual(shape, K, offset, nq=(2, 3), hessian=True, seed=0):
    """A field with sparse seed derivatives, and the same with dense."""
    from skfem.autodiff.dual import Dual
    from skfem.element.block_array import BlockArray
    rng = np.random.RandomState(seed)
    val = rng.rand(*(shape + nq))
    ga = BlockArray((K,) + shape + (1, 1),
                    1 + len(shape),
                    {(offset + j,) + ix: np.ones((1, 1))
                     for j, ix in enumerate(np.ndindex(*shape))})
    sparse = Dual(val, ga=ga, gb=ga if hessian else None)
    dense = Dual(val,
                 ga=np.asarray(ga),
                 gb=np.asarray(ga) if hessian else None)
    return sparse, dense


@pytest.mark.parametrize("expr", [
    lambda u, G: ddot(G, G) * dot(u, u),
    lambda u, G: trace(mul(transpose(G), G)) ** 2,
    lambda u, G: det(G + identity(G)),
    lambda u, G: np.sum(inv(G + 3. * identity(G)), axis=(0, 1)),
    lambda u, G: np.sum(G * G, axis=0)[1] * u[2],
    lambda u, G: dot(cross(u, G[:, 0]), u),
    lambda u, G: dot(mul(G, u), u) / (1. + dot(u, u)),
    lambda u, G: ddot(sym_grad_like(G), prod(u, u)),
    lambda u, G: np.sum(stack([u[0] * G[1, 1], u[2], 2. * u[1]], axis=0)
                        * u, axis=0),
    lambda u, G: np.where(u[0] > .5, u[0] ** 3, np.exp(u[1])) * G[0, 2],
    lambda u, G: np.maximum(trace(G), 1.) * np.sqrt(1. + dot(u, u)),
    lambda u, G: np.trace(G, axis1=0, axis2=1) * u[0],
    lambda u, G: ddot(G.T.T, eye(u[0], 3)) * u[1],  # full transpose
    lambda u, G: np.sum(G[0:2, 1] * u[:2], axis=0),  # slicing
    lambda u, G: np.einsum('i...,i...', u, u) * np.einsum('ii...', G),
])
def test_sparse_dual_matches_dense(expr):
    """Sparse seed derivatives give the same result as dense ones."""
    from skfem.element.block_array import BlockArray
    K = 12
    u, ud = _sparse_dual((3,), K, 0, seed=0)
    G, Gd = _sparse_dual((3, 3), K, 3, seed=1)
    out, ref = expr(u, G), expr(ud, Gd)
    assert_allclose(out.val, ref.val)
    for attr in ('ga', 'gb', 'hab'):
        d, r = getattr(out, attr), getattr(ref, attr)
        assert (d is None) == (r is None)
        if d is not None:
            assert_allclose(np.broadcast_to(np.asarray(d), r.shape), r,
                            atol=1e-13)
    assert isinstance(out.hab, (BlockArray, np.ndarray))


def sym_grad_like(G):
    return .5 * (G + transpose(G))


def test_sparse_dual_stays_sparse():
    from skfem.element.block_array import BlockArray
    K = 12
    u, _ = _sparse_dual((3,), K, 0)
    G, _ = _sparse_dual((3, 3), K, 3)
    F = G + identity(G)
    E = .5 * (mul(transpose(F), F) - identity(G))
    W = ddot(E, E) + trace(E) ** 2 + dot(u, u)
    assert isinstance(E.hab, BlockArray)
    assert isinstance(W.hab, BlockArray)
    # the energy depends on u and grad(u) but has no mixed derivatives
    assert len(W.hab.blocks) == 9 * 9 + 3


def test_vector_p2_finite_difference():
    basis = Basis(MeshTet(), ElementVector(ElementTetP2()))

    def form(u, v, w):
        G = grad(u)
        return (ddot(G + G.T.T, grad(v)) * (1. + np.sum(G * G, axis=(0, 1)))
                + dot(stack([u[1] * u[2], u[0], np.exp(u[2])]), v)
                + np.where(u[0] > .03, u[0], 0.) * np.einsum('ii...', G)
                * v[1])

    x = .05 * np.random.RandomState(0).rand(basis.N)
    _fd_check(form, basis, x)
