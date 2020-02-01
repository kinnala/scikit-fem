import unittest

import numpy as np

from skfem import *
from skfem.models.poisson import laplace


class ConvergenceRaviartThomas(unittest.TestCase):
    rateL2 = 1.0
    rateHdiv = 1.0
    eps = 0.1

    def runTest(self):

        @BilinearForm
        def bilinf_A(sigma, tau, w):
            from skfem.models.helpers import dot
            return dot(sigma, tau)

        @BilinearForm
        def bilinf_B(sigma, v, w):
            return sigma.df * v

        @LinearForm
        def load(v, w):
            x = w.x
            if x.shape[0] == 1:
                return (np.sin(np.pi * x[0]) * (np.pi ** 2) * v)
            elif x.shape[0] == 2:
                return (np.sin(np.pi * x[0]) *
                        np.sin(np.pi * x[1]) * (2.0 * np.pi ** 2) * v)
            elif x.shape[0] == 3:
                return (np.sin(np.pi * x[0]) *
                        np.sin(np.pi * x[1]) *
                        np.sin(np.pi * x[2]) * (3.0 * np.pi ** 2) * v)
            else:
                raise Exception("The dimension not supported")

        m = self.mesh
        Nitrs = 3
        L2s = np.zeros(Nitrs)
        Hdivs = np.zeros(Nitrs)
        hs = np.zeros(Nitrs)

        for itr in range(Nitrs):
            ib1, ib2 = self.create_basis(m)

            A = asm(bilinf_A, ib1)
            B = asm(bilinf_B, ib1, ib2)
            b = np.concatenate((
                np.zeros(A.shape[0]),
                -asm(load, ib2)
            ))

            from scipy.sparse import bmat
            K = bmat([[A, B.T], [B, None]]).tocsr()

            x = solve(K, b)

            # calculate error
            sigma, u = np.split(x, [A.shape[0]])

            L2s[itr] = self.compute_L2(m, ib2, u)
            Hdivs[itr] = self.compute_Hdiv(m, ib1, sigma)
            hs[itr] = m.param()

            m.refine()

        rateL2 = np.polyfit(np.log(hs), np.log(L2s), 1)[0]
        rateHdiv = np.polyfit(np.log(hs), np.log(Hdivs), 1)[0]

        self.assertLess(np.abs(rateL2 - self.rateL2),
                        self.eps,
                        msg='observed L2 rate: {}'.format(rateL2))
        self.assertLess(np.abs(rateHdiv - self.rateHdiv),
                        self.eps,
                        msg='observed Hdiv rate: {}'.format(rateHdiv))
        self.assertLess(Hdivs[-1], 0.04)
        self.assertLess(L2s[-1], 0.01)

    def compute_L2(self, m, basis, U):
        uh, *_ = basis.interpolate(U)
        dx = basis.dx
        x = basis.global_coordinates()

        def u(y):
            if y.shape[0] == 1:
                return np.sin(np.pi * y[0])
            elif y.shape[0] == 2:
                return (np.sin(np.pi * y[0]) *
                        np.sin(np.pi * y[1]))
            return (np.sin(np.pi * y[0]) *
                    np.sin(np.pi * y[1]) *
                    np.sin(np.pi * y[2]))

        return np.sqrt(np.sum(np.sum((uh - u(x.f)) ** 2 * dx, axis=1)))

    def compute_Hdiv(self, m, basis, U):
        uh, duh, *_ = basis.interpolate(U)
        dx = basis.dx
        x = basis.global_coordinates()

        def divu(y):
            if y.shape[0] == 1:
                return np.pi * np.cos(np.pi * y[0])
            elif y.shape[0] == 2:
                return (np.pi * np.cos(np.pi * y[0]) * np.sin(np.pi * y[1]) +
                        np.pi * np.sin(np.pi * y[0]) * np.cos(np.pi * y[1]))
            raise Exception("!")

        return np.sqrt(np.sum(np.sum(((uh[0] + uh[1] - divu(x.f)) ** 2) * dx, axis=1)))

        return np.sqrt(np.sum(np.sum((uh - u(x.f)) ** 2 * dx, axis=1)))

    def create_basis(self, m):
        e = ElementTriRT0()
        e0 = ElementTriP0()
        return (InteriorBasis(m, e, intorder=2),
                InteriorBasis(m, e0, intorder=2))

    def setUp(self):
        self.mesh = MeshTri()
        self.mesh.refine(4)
