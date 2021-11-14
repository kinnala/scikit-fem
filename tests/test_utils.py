from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal

from skfem.assembly import InteriorBasis
from skfem.element import ElementTriP1
from skfem.mesh import MeshTri
from skfem.utils import projection, enforce
from skfem.models import laplace, mass


class InitializeScalarField(TestCase):

    def runTest(self):

        mesh = MeshTri().refined(5)
        basis = InteriorBasis(mesh, ElementTriP1())

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
        basis = InteriorBasis(m, e)

        A = laplace.assemble(basis)
        M = mass.assemble(basis)
        D = m.boundary_nodes()

        assert_almost_equal(enforce(A, D=D).toarray(), np.eye(A.shape[0]))
        assert_almost_equal(enforce(M, D=D, diag=0.).toarray(),
                            np.zeros(M.shape))

        enforce(A, D=D, overwrite=True)
        assert_almost_equal(A.toarray(), np.eye(A.shape[0]))
