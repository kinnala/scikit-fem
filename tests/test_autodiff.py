import pytest
import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal
from skfem.experimental.autodiff import NonlinearForm
from skfem.experimental.autodiff.helpers import grad, dot
from skfem.assembly import Basis
from skfem.mesh import MeshTri
from skfem.element import ElementTriP1
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
