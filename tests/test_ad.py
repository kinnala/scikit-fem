import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal,
                           assert_allclose)

from skfem import (Basis, FacetBasis, MeshTri, MeshQuad, MeshTet, MeshLine,
                   ElementTriP1, ElementTriP2, ElementQuad1, ElementTetP1,
                   ElementVector, solve, condense, enforce, bmat)
from skfem.ad import NonlinearForm
import skfem.ad.helpers as ngh
from skfem.ad.helpers import (grad, dot, ddot, mul, div, sym_grad,
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


def test_minimal_surface():
    """Same as docs/examples/ex45.py."""

    m = MeshTri().refined(5)

    @NonlinearForm(hessian=True)
    def energy(u, _):
        return np.sqrt(1. + dot(grad(u), grad(u)))

    basis = Basis(m, ElementTriP1())

    D = m.boundary_nodes()
    x = np.sin(np.pi * m.p[0])

    for itr in range(100):
        x_prev = x.copy()
        x += solve(*condense(*energy.assemble(basis, x=x), D=D))
        if np.linalg.norm(x - x_prev) < 1e-8:
            break

    assert_almost_equal(np.mean(x), 0.277931521728906)


def test_contact():
    """Same as docs/examples/ex51.py."""
    pytest.importorskip('shapely')
    from skfem.supermeshing import intersect, elementwise_quadrature

    m1 = (MeshQuad
          .init_tensor(np.linspace(0, 5, 60), np.linspace(0, 0.25, 5))
          .with_defaults())
    m2 = (MeshQuad
          .init_tensor(np.linspace(0, 5, 51), np.linspace(-0.5, -0.25, 4))
          .with_defaults())

    e1 = ElementVector(ElementQuad1())
    e2 = ElementVector(ElementQuad1())

    basis1 = Basis(m1, e1)
    basis2 = Basis(m2, e2)

    @NonlinearForm(hessian=True)
    def g(u1, u2, w):
        eps = 1e-2
        return 1 / eps * np.minimum(0.25 + u1[1] - u2[1], 0) ** 2

    @NonlinearForm(hessian=True)
    def J1(u, w):
        eps = 1/2 * (grad(u) + transpose(grad(u))
                     + mul(transpose(grad(u)), grad(u)))
        sig = 2 * eps + eye(trace(eps), 2)
        return 1/2 * ddot(sig, eps)

    @NonlinearForm(hessian=True)
    def J2(u, w):
        eps = 1/2 * (grad(u) + transpose(grad(u))
                     + mul(transpose(grad(u)), grad(u)))
        sig = 2 * eps + eye(trace(eps), 2)
        return 1/2 * ddot(sig, eps) - 1e-2 * u[1] - 5e-2 * u[0]

    x = np.zeros(basis1.N + basis2.N)
    m1defo = m1.copy()
    m2defo = m2.copy()

    for itr in range(40):
        m1t, orig1 = m1defo.trace('bottom',
                                  mtype=MeshLine,
                                  project=lambda p: np.array(p[0]))
        m2t, orig2 = m2defo.trace('top',
                                  mtype=MeshLine,
                                  project=lambda p: np.array(p[0]))
        m12, t1, t2 = intersect(m1t, m2t)

        fbasis1 = FacetBasis(m1, e1,
                             quadrature=elementwise_quadrature(m1t, m12, t1),
                             facets=orig1[t1])
        fbasis2 = FacetBasis(m2, e2,
                             quadrature=elementwise_quadrature(m2t, m12, t2),
                             facets=orig2[t2])
        fbasis = fbasis1 * fbasis2

        jac1, rhs1 = J1.assemble(basis1, x=x[:basis1.N])
        jac2, rhs2 = J2.assemble(basis2, x=x[basis1.N:])
        jac = bmat([[jac1, None],
                    [None, jac2]])
        rhs = np.concatenate((rhs1, rhs2))

        jacg, rhsg = g.assemble(fbasis, x=x)
        jac = jac + jacg
        rhs = rhs + rhsg

        dx = solve(*enforce(jac, np.array(rhs), D=np.concatenate((
            basis1.get_dofs({'left', 'right'}).all(),
            basis2.get_dofs({'left', 'right'}).all() + basis1.N,
        ))))
        x += 0.95 * dx
        if np.linalg.norm(dx) < 1e-8:
            break
        (x1, _), (x2, _) = fbasis.split(x)
        m1defo = m1.translated(x1[basis1.nodal_dofs])
        m2defo = m2.translated(x2[basis2.nodal_dofs])

    assert_almost_equal(x1.max(), 0.15983570088457588)
    assert_almost_equal(x2.max(), 0.4062806125571157)

    # helpers work also outside of forms
    u1 = basis1.interpolate(x1)
    eps1 = 1/2 * (grad(u1) + transpose(grad(u1))
                  + mul(transpose(grad(u1)), grad(u1)))
    sig1 = 2 * eps1 + eye(trace(eps1), 2)
    assert sig1.shape == grad(u1).shape


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


def test_compare_to_jax():
    pytest.importorskip('jax')
    import jax.numpy as jnp
    import skfem.autodiff as ad
    import skfem.autodiff.helpers as adh

    basis = Basis(MeshTri().refined(2),
                  ElementVector(ElementTriP2()) * ElementTriP1())

    def make(h, xp):
        def form(u, p, v, q, w):
            P = p * 1.
            return (h.ddot(h.sym_grad(u), h.sym_grad(v))
                    + h.dot(h.mul(h.grad(u), u), v)
                    + xp.exp(P) * h.dot(u, h.grad(q))
                    - h.div(u) * q - h.div(v) * P ** 3)
        return form

    x = .1 * np.random.RandomState(0).rand(basis.N)
    jac1, rhs1 = ad.NonlinearForm(make(adh, jnp)).assemble(basis, x=x)
    jac2, rhs2 = NonlinearForm(make(ngh, np)).assemble(basis, x=x)
    assert_allclose(jac1.toarray(), jac2.toarray(), atol=1e-10)
    assert_allclose(rhs1, rhs2, atol=1e-10)
