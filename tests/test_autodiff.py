import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal,
                           assert_allclose)

from skfem import (Basis, FacetBasis, MeshTri, MeshQuad, MeshTet, MeshLine,
                   ElementTriP1, ElementTriP2, ElementQuad1, ElementTetP1,
                   ElementVector, solve, condense, enforce, bmat)
from skfem.autodiff import NonlinearForm
import skfem.autodiff.helpers as ngh
from skfem.autodiff.helpers import (grad, dot, ddot, mul, div, sym_grad,
                                    transpose, eye, trace, det, inv,
                                    cross, prod, identity, inner)


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
