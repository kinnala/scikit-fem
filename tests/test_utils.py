from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal

from skfem.assembly import CellBasis
from skfem.element import ElementTriP1
from skfem.mesh import MeshTri
from skfem.utils import projection, enforce, condense, solve
from skfem.models import laplace, mass, unit_load


class InitializeScalarField(TestCase):

    def runTest(self):

        mesh = MeshTri().refined(5)
        basis = CellBasis(mesh, ElementTriP1())

        def fun(X):
            x, y = X
            return x ** 2 + y ** 2

        x = projection(fun, basis)
        y = fun(mesh.p)

        normest = np.linalg.norm(x - y)

        self.assertTrue(normest < 0.011,
                        msg="|x-y| = {}".format(normest))


class TestEnforce(TestCase):

    mesh = MeshTri()

    def runTest(self):

        m = self.mesh
        e = ElementTriP1()
        basis = CellBasis(m, e)

        A = laplace.assemble(basis)
        M = mass.assemble(basis)
        D = m.boundary_nodes()

        assert_almost_equal(enforce(A, D=D).toarray(), np.eye(A.shape[0]))
        assert_almost_equal(enforce(M, D=D, diag=0.).toarray(),
                            np.zeros(M.shape))

        enforce(A, D=D, overwrite=True)
        assert_almost_equal(A.toarray(), np.eye(A.shape[0]))


def test_simple_cg_solver():

    m = MeshTri().refined(3)
    basis = CellBasis(m, ElementTriP1())

    A0 = laplace.coo_data(basis)
    A1 = laplace.assemble(basis)

    f = unit_load.assemble(basis)

    D = m.boundary_nodes()

    x1 = solve(*condense(A1, f, D=D))

    f[D] = 0

    x0 = A0.solve(f, D=D)

    assert_almost_equal(x0, x1)
