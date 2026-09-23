import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfem import Basis, ElementTriHermite, MeshTri1DG
from skfem.models.poisson import laplace, mass


def periodic_mesh(periodic, irregular):
    if irregular:
        x = np.array([0., .1, .35, .7, 1.])
        y = np.array([0., .2, .4, .65, 1.])
    else:
        x = np.linspace(0., 1., 5)
        y = np.linspace(0., 1., 5)
    return MeshTri1DG.init_tensor(x + (1.5 if irregular else 0.),
                                  y - (.75 if irregular else 0.),
                                  periodic=periodic)


@pytest.mark.parametrize('periodic', [[0], [1], [0, 1]])
@pytest.mark.parametrize('irregular', [False, True])
def test_periodic_tri_hermite_assembly(periodic, irregular):
    mesh = periodic_mesh(periodic, irregular)
    ordinary = Basis(mesh._orig, ElementTriHermite(), intorder=6)
    basis = Basis(mesh, ElementTriHermite(), intorder=6)
    restriction = np.zeros((ordinary.N, basis.N))
    for vertex, identified in enumerate(mesh._ix):
        restriction[ordinary.nodal_dofs[:, vertex],
                    basis.nodal_dofs[:, identified]] = 1.
    restriction[ordinary.interior_dofs.ravel(),
                basis.interior_dofs.ravel()] = 1.
    for form in (mass, laplace):
        expected = restriction.T @ form.assemble(ordinary) @ restriction
        assert_allclose(form.assemble(basis).toarray(), expected,
                        rtol=1e-8, atol=1e-9)

    # The three nodal DOFs agree at identified vertices.  This is not a
    # claim of C1 continuity at every point along a triangle edge.
    vertices = Basis(mesh, ElementTriHermite(),
                     quadrature=(ElementTriHermite.refdom.p, np.ones(3)))
    coefficients = np.random.default_rng(902).normal(size=basis.N)
    field = vertices.interpolate(coefficients)
    for index, values in enumerate((field, field.grad[0], field.grad[1])):
        expected = coefficients[basis.nodal_dofs[index, mesh.t]].T
        assert_allclose(values, expected, rtol=1e-8, atol=1e-8)


def cubic(x, y):
    return 1. + 2.*x - 3.*y + x*y + .2*x**3 + .4*y**3 - .7*x*x*y + .6*x*y*y


def cubic_gradient(x, y):
    return np.array([2. + y + .6*x*x - 1.4*x*y + .6*y*y,
                     -3. + x + 1.2*y*y - .7*x*x + 1.2*x*y])


@pytest.mark.parametrize('periodic', [[0], [1], [0, 1]])
@pytest.mark.parametrize('irregular', [False, True])
def test_periodic_tri_hermite_local_polynomial(periodic, irregular):
    mesh = periodic_mesh(periodic, irregular)
    basis = Basis(mesh, ElementTriHermite(), intorder=6)
    vertices = mesh.mapping().F(ElementTriHermite.refdom.p)
    coefficients = []
    for vertex in range(3):
        x, y = vertices[:, :, vertex]
        coefficients.extend([cubic(x, y), *cubic_gradient(x, y)])
    x, y = vertices.mean(axis=2)
    coefficients.append(cubic(x, y))
    # Element-local coefficients avoid imposing a nonperiodic polynomial
    # on globally identified nodal DOFs.
    value = np.zeros_like(basis.dx)
    gradient = np.zeros((2, *basis.dx.shape))
    for coefficient, (shape,) in zip(coefficients, basis.basis):
        value += coefficient[:, None] * shape
        gradient += coefficient[None, :, None] * shape.grad
    x, y = basis.global_coordinates()
    assert_allclose(value, cubic(x, y), rtol=1e-8, atol=1e-9)
    assert_allclose(gradient, cubic_gradient(x, y), rtol=1e-8, atol=1e-8)
