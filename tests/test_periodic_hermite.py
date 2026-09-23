import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfem import Basis, ElementLineHermite, MeshLine, MeshLine1DG
from skfem.models.poisson import laplace, mass


@pytest.mark.parametrize('offset', [0., 1.5])
@pytest.mark.parametrize('points', [
    np.linspace(0., 1., 10),
    np.array([0., .1, .3, .55, .8, 1.]),
])
def test_periodic_hermite_assembly(points, offset):
    mesh = MeshLine(points + offset)
    periodic = MeshLine1DG.periodic(mesh, [len(points) - 1], [0])
    ordinary = Basis(mesh, ElementLineHermite())
    basis = Basis(periodic, ElementLineHermite())

    # Identify both value and derivative DOFs at the ordinary mesh endpoints.
    restriction = np.zeros((ordinary.N, basis.N))
    for node in range(len(points)):
        restriction[ordinary.nodal_dofs[:, node],
                    basis.nodal_dofs[:, node % (len(points) - 1)]] = 1.
    for form in (mass, laplace):
        expected = restriction.T @ form.assemble(ordinary) @ restriction
        assert_allclose(form.assemble(basis).toarray(), expected,
                        rtol=1e-8, atol=1e-9)

    # Arbitrary coefficients must give a C1-periodic function, including
    # across the seam where physical coordinates differ by the period.
    endpoints = Basis(periodic, ElementLineHermite(),
                      quadrature=(np.array([[0., 1.]]), np.ones(2)))
    field = endpoints.interpolate(np.random.default_rng(902).normal(size=basis.N))
    for values in (field, field.grad[0]):
        assert_allclose(values[:, 1], np.roll(values[:, 0], -1),
                        rtol=1e-8, atol=1e-8)


def test_periodic_hermite_convergence():
    errors = []
    for n in (8, 16, 32):
        mesh = MeshLine(np.linspace(0., 1., n + 1))
        periodic = MeshLine1DG.periodic(mesh, [n], [0])
        basis = Basis(periodic, ElementLineHermite(), intorder=8)
        coefficients = np.zeros(basis.N)
        nodes = np.linspace(0., 1., n + 1)[:-1]
        coefficients[basis.nodal_dofs[0]] = np.sin(2. * np.pi * nodes)
        coefficients[basis.nodal_dofs[1]] = 2. * np.pi * np.cos(2. * np.pi * nodes)
        field = basis.interpolate(coefficients)
        x = basis.global_coordinates()[0]
        errors.append([
            np.sqrt(np.sum((field - np.sin(2. * np.pi * x)) ** 2 * basis.dx)),
            np.sqrt(np.sum((field.grad[0] - 2. * np.pi * np.cos(2. * np.pi * x)) ** 2
                           * basis.dx)),
        ])
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert_allclose(rates, np.tile([4., 3.], (2, 1)), atol=.15)
