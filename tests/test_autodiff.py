import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal)

try:
    import jax.numpy as jnp
    from skfem.experimental.autodiff import NonlinearForm
    from skfem.experimental.autodiff.helpers import (grad, dot,
                                                     ddot, mul,
                                                     div, sym_grad,
                                                     transpose,
                                                     eye, trace)
except Exception:
    def NonlinearForm(x):
        raise Exception("jax failed to import")

from skfem.assembly import Basis
from skfem.mesh import MeshTri, MeshQuad
from skfem.element import (ElementTriP1, ElementTriP2,
                           ElementVector)
from skfem.utils import solve, condense


def test_linear_poisson():

    m = MeshTri().refined(4)
    basis = Basis(m, ElementTriP1())
    x = basis.zeros()
    xe = basis.project(lambda x: (1 / (2. * np.pi ** 2)
                                  * (np.sin(np.pi * x[0])
                                     * np.sin(np.pi * x[1]))))

    @NonlinearForm
    def poisson(u, v, w):
        f = jnp.sin(jnp.pi * w.x[0]) * jnp.sin(jnp.pi * w.x[1])
        return dot(grad(u), grad(v)) - f * v

    for itr in range(10):
        xp = x.copy()
        x += solve(*condense(*poisson.assemble(basis, x=x),
                             D=basis.get_dofs()))
        if jnp.linalg.norm(x - xp) < 1e-8:
            break

    assert_array_almost_equal(x, xe, decimal=3)


def test_obstacle_problem():

    m = MeshTri().refined(3)
    basis = Basis(m, ElementTriP1())
    x = basis.zeros()

    @NonlinearForm
    def poisson(u, v, w):
        f = jnp.sin(jnp.pi * w.x[0]) * jnp.sin(jnp.pi * w.x[1])
        return (dot(grad(u), grad(v))
                + 1e5 * jnp.maximum(u - 0.01, 0) * v
                - f * v)

    for itr in range(10):
        xp = x.copy()
        x += solve(*condense(*poisson.assemble(basis, x=x),
                             D=basis.get_dofs()))
        if jnp.linalg.norm(x - xp) < 1e-8:
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
        res = jnp.linalg.norm(x - xp)
        print(res)
        if res < 1e-8:
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
        res = jnp.linalg.norm(x - xp)
        print(res)
        if res < 1e-8:
            break

    assert_almost_equal(np.max(x), 2.83411524813795)
