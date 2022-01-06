import unittest

import numpy as np

from skfem import (MeshLine, ElementLinePp, InteriorBasis, LinearForm, asm,
                   solve, condense, MeshQuad, ElementQuadP)

from skfem.models.poisson import laplace


class ConvergenceLinePp(unittest.TestCase):
    """Solve up to floating point precision."""

    def setUp(self):
        self.mesh = MeshLine().refined(3)

    def create_basis(self, m, p):
        e = ElementLinePp(p)
        return InteriorBasis(m, e)

    def runTest(self):

        @LinearForm
        def load(v, w):
            x = w.x
            return np.sin(np.pi * x[0]) * (np.pi ** 2) * v

        m = self.mesh
        Nitrs = 8
        L2s = np.zeros(Nitrs)
        H1s = np.zeros(Nitrs)

        for itr in range(Nitrs):
            ib = self.create_basis(m, itr + 2)

            A = asm(laplace, ib)
            b = asm(load, ib)

            D = ib.get_dofs()

            x = solve(*condense(A, b, D=D))

            # calculate error
            L2s[itr], H1s[itr] = self.compute_error(m, ib, x)

        self.assertLess(H1s[-1], 1e-13)
        self.assertLess(L2s[-1], 1e-13)

    def compute_error(self, m, basis, U):
        uh, duh, *_ = basis.interpolate(U).astuple
        dx = basis.dx
        x = basis.global_coordinates()

        def u(y):
            return np.sin(np.pi * y[0])

        L2 = np.sqrt(np.sum(np.sum((uh - u(x.value)) ** 2 * dx, axis=1)))

        def ux(y):
            return np.pi * np.cos(np.pi * y[0])

        H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x.value)) ** 2) * dx, axis=1)))

        return L2, H1


class ConvergenceQuadP(unittest.TestCase):
    """Solve up to floating point precision."""

    def setUp(self):
        self.mesh = MeshQuad().refined(1)
        self.mesh.p[:, self.mesh.interior_nodes()] +=\
            0.05 * self.mesh.p[:, self.mesh.interior_nodes()]

    def create_basis(self, m, p):
        e = ElementQuadP(p)
        return InteriorBasis(m, e, intorder=p*p)

    def runTest(self):

        @LinearForm
        def load(v, w):
            x = w.x
            return (np.sin(np.pi * x[0]) *
                    np.sin(np.pi * x[1]) * (2.0 * np.pi ** 2) * v)

        m = self.mesh
        Nitrs = 9
        L2s = np.zeros(Nitrs)
        H1s = np.zeros(Nitrs)

        for itr in range(Nitrs):
            ib = self.create_basis(m, itr + 2)

            A = asm(laplace, ib)
            b = asm(load, ib)

            D = ib.get_dofs()

            x = solve(*condense(A, b, D=D))

            # calculate error
            L2s[itr], H1s[itr] = self.compute_error(m, ib, x)

        self.assertLess(H1s[-1], 1e-10)
        self.assertLess(L2s[-1], 1e-11)

    def compute_error(self, m, basis, U):
        uh, duh, *_ = basis.interpolate(U).astuple
        dx = basis.dx
        x = basis.global_coordinates()

        def u(y):
            return (np.sin(np.pi * y[0]) *
                    np.sin(np.pi * y[1]))

        L2 = np.sqrt(np.sum(np.sum((uh - u(x.value)) ** 2 * dx, axis=1)))

        def ux(y):
            return np.pi * (np.cos(np.pi * y[0]) *
                            np.sin(np.pi * y[1]))

        H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x.value)) ** 2) * dx, axis=1)))

        return L2, H1
