import unittest

import numpy as np

from skfem import BilinearForm, LinearForm, asm, solve, condense
from skfem.models.poisson import laplace
from skfem.element import (
    ElementHex1,
    ElementHex2,
    ElementHexS2,
    ElementLineP1,
    ElementLineP2,
    ElementLineMini,
    ElementTetCR,
    ElementQuad1,
    ElementQuad2,
    ElementQuadS2,
    ElementTetMini,
    ElementTetP1,
    ElementTetP2,
    ElementTriMini,
    ElementTriCCR,
    ElementTriP1,
    ElementTriP2,
    ElementTriCR,
    ElementTriHermite,
    ElementTetCCR)
from skfem.assembly import FacetBasis, InteriorBasis
from skfem.mesh import MeshHex, MeshLine, MeshQuad, MeshTet, MeshTri


class ConvergenceQ1(unittest.TestCase):

    rateL2 = 2.0
    rateH1 = 1.0
    eps = 0.1

    def runTest(self):

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
        H1s = np.zeros(Nitrs)
        hs = np.zeros(Nitrs)

        for itr in range(Nitrs):
            if itr > 0:
                m = m.refined()
            ib = self.create_basis(m)

            A = asm(laplace, ib)
            b = asm(load, ib)

            D = self.get_bc_nodes(ib)
            x = solve(*condense(A, b, D=D))

            # calculate error
            L2s[itr], H1s[itr] = self.compute_error(m, ib, x)
            hs[itr] = m.param()

        rateL2 = np.polyfit(np.log(hs), np.log(L2s), 1)[0]
        rateH1 = np.polyfit(np.log(hs), np.log(H1s), 1)[0]

        self.assertLess(np.abs(rateL2 - self.rateL2),
                        self.eps,
                        msg='observed L2 rate: {}'.format(rateL2))
        self.assertLess(np.abs(rateH1 - self.rateH1),
                        self.eps,
                        msg='observed H1 rate: {}'.format(rateH1))
        self.assertLess(H1s[-1], 0.3)
        self.assertLess(L2s[-1], 0.008)

    def get_bc_nodes(self, ib):
        return ib.get_dofs().all('u')

    def compute_error(self, m, basis, U):
        uh, duh, *_ = basis.interpolate(U)
        dx = basis.dx
        x = basis.global_coordinates()

        def u(y):
            if y.shape[0] == 1:
                return np.sin(np.pi * y[0])
            elif y.shape[0] == 2:
                return (np.sin(np.pi * y[0]) *
                        np.sin(np.pi * y[1]))
            elif y.shape[0] == 3:
                return (np.sin(np.pi * y[0]) *
                        np.sin(np.pi * y[1]) *
                        np.sin(np.pi * y[2]))
            else:
                raise Exception("The dimension not supported")

        L2 = np.sqrt(np.sum(np.sum((uh - u(x.value)) ** 2 * dx, axis=1)))

        def ux(y):
            if y.shape[0] == 1:
                return np.pi * np.cos(np.pi * y[0])
            elif y.shape[0] == 2:
                return np.pi * (np.cos(np.pi * y[0]) *
                                np.sin(np.pi * y[1]))
            elif y.shape[0] == 3:
                return np.pi * (np.cos(np.pi * y[0]) *
                                np.sin(np.pi * y[1]) *
                                np.sin(np.pi * y[2]))
            else:
                raise Exception("The dimension not supported")

        if m.dim() >= 2:
            def uy(y):
                if y.shape[0] == 2:
                    return np.pi * (np.sin(np.pi * y[0]) *
                                    np.cos(np.pi * y[1]))
                elif y.shape[0] == 3:
                    return np.pi * (np.sin(np.pi * y[0]) *
                                    np.cos(np.pi * y[1]) *
                                    np.sin(np.pi * y[2]))
                else:
                    raise Exception("The dimension not supported")

        if m.dim() == 3:
            def uz(y):
                return np.pi * (np.sin(np.pi * y[0]) *
                                np.sin(np.pi * y[1]) *
                                np.cos(np.pi * y[2]))

        if m.dim() == 3:
            H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x.value)) ** 2 +
                                        (duh[1] - uy(x.value)) ** 2 +
                                        (duh[2] - uz(x.value)) ** 2) * dx,
                                axis=1)))
        elif m.dim() == 2:
            H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x.value)) ** 2 +
                                        (duh[1] - uy(x.value)) ** 2) * dx,
                                axis=1)))
        else:
            H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x.value)) ** 2) * dx,
                                axis=1)))

        return L2, H1

    def create_basis(self, m):
        e = ElementQuad1()
        return InteriorBasis(m, e)

    def setUp(self):
        self.mesh = MeshQuad().refined(2)


class ConvergenceQ2(ConvergenceQ1):

    rateL2 = 3.0
    rateH1 = 2.0

    def create_basis(self, m):
        e = ElementQuad2()
        return InteriorBasis(m, e)

    def setUp(self):
        self.mesh = MeshQuad().refined(2)


class ConvergenceQuadS2(ConvergenceQ2):

    def create_basis(self, m):
        e = ElementQuadS2()
        return InteriorBasis(m, e)


class ConvergenceTriP1(ConvergenceQ1):

    def create_basis(self, m):
        e = ElementTriP1()
        return InteriorBasis(m, e)

    def setUp(self):
        self.mesh = MeshTri.init_sqsymmetric().refined(2)


class ConvergenceTriP2(ConvergenceTriP1):

    rateL2 = 3.0
    rateH1 = 2.0

    def create_basis(self, m):
        e = ElementTriP2()
        return InteriorBasis(m, e)


class ConvergenceTriHermite(ConvergenceTriP1):

    rateL2 = 4.0
    rateH1 = 3.0
    eps = 0.15

    def create_basis(self, m):
        e = ElementTriHermite()
        return InteriorBasis(m, e)

    def get_bc_nodes(self, ib):
        return np.concatenate((
            ib.get_dofs().all('u'),
            ib.get_dofs(lambda x: x[0] == 0).all('u_y'),
            ib.get_dofs(lambda x: x[0] == 1).all('u_y'),
            ib.get_dofs(lambda x: x[1] == 0).all('u_x'),
            ib.get_dofs(lambda x: x[1] == 1).all('u_x'),
        ))

    def setUp(self):
        self.mesh = MeshTri.init_symmetric().refined(2)


class ConvergenceTriCR(ConvergenceTriP1):

    rateL2 = 2.0
    rateH1 = 1.0

    def create_basis(self, m):
        e = ElementTriCR()
        return InteriorBasis(m, e)


class ConvergenceTriCCR(ConvergenceTriP1):

    rateL2 = 3.0
    rateH1 = 2.0

    def create_basis(self, m):
        e = ElementTriCCR()
        return InteriorBasis(m, e)


class ConvergenceTriMini(ConvergenceTriP1):

    def create_basis(self, m):
        e = ElementTriMini()
        return InteriorBasis(m, e)


class ConvergenceHex1(ConvergenceQ1):

    rateL2 = 2.0
    rateH1 = 1.0
    eps = 0.11

    def create_basis(self, m):
        e = ElementHex1()
        return InteriorBasis(m, e)

    def setUp(self):
        self.mesh = MeshHex().refined(2)


class ConvergenceHexSplitTet1(ConvergenceQ1):

    rateL2 = 2.0
    rateH1 = 1.0
    eps = 0.18

    def create_basis(self, m):
        e = ElementTetP1()
        return InteriorBasis(m, e)

    def setUp(self):
        self.mesh = MeshHex().refined(2).to_meshtet()


class ConvergenceHexS2(ConvergenceQ1):

    rateL2 = 3.05
    rateH1 = 2.21
    eps = 0.02

    def create_basis(self, m):
        e = ElementHexS2()
        return InteriorBasis(m, e)

    def setUp(self):
        self.mesh = MeshHex().refined(1)


class ConvergenceHex2(ConvergenceHexS2):

    rateL2 = 2.92
    rateH1 = 2.01

    def create_basis(self, m):
        e = ElementHex2()
        return InteriorBasis(m, e)


class ConvergenceTetP1(ConvergenceQ1):

    rateL2 = 2.0
    rateH1 = 1.0
    eps = 0.13

    def create_basis(self, m):
        e = ElementTetP1()
        return InteriorBasis(m, e)

    def setUp(self):
        self.mesh = MeshTet().refined(2)


class ConvergenceTetCR(ConvergenceTetP1):

    rateL2 = 2.2
    rateH1 = 1.13

    def create_basis(self, m):
        e = ElementTetCR()
        return InteriorBasis(m, e)


class ConvergenceTetCCR(ConvergenceTetP1):

    rateL2 = 3.0
    rateH1 = 2.0

    def create_basis(self, m):
        e = ElementTetCCR()
        return InteriorBasis(m, e)


class ConvergenceTetP2(ConvergenceTetP1):

    rateL2 = 3.23
    rateH1 = 1.94
    eps = 0.01

    def create_basis(self, m):
        e = ElementTetP2()
        return InteriorBasis(m, e)

    def setUp(self):
        self.mesh = MeshTet().refined(1)


class ConvergenceTetMini(ConvergenceTetP1):

    def create_basis(self, m):
        e = ElementTetMini()
        return InteriorBasis(m, e, intorder=3)


class ConvergenceLineP1(ConvergenceQ1):

    def create_basis(self, m):
        e = ElementLineP1()
        return InteriorBasis(m, e)

    def setUp(self):
        self.mesh = MeshLine().refined(3)


class ConvergenceLineP2(ConvergenceQ1):

    rateL2 = 3.0
    rateH1 = 2.0

    def create_basis(self, m):
        e = ElementLineP2()
        return InteriorBasis(m, e)

    def setUp(self):
        self.mesh = MeshLine().refined(3)


class ConvergenceLineMini(ConvergenceLineP2):

    def create_basis(self, m):
        e = ElementLineMini()
        return InteriorBasis(m, e)


class FacetConvergenceTetP2(unittest.TestCase):
    """Test second order tetrahedral element and facet
    assembly."""
    case = (MeshTet, ElementTetP2)
    limits = (1.9, 2.2)
    preref = 1

    def runTest(self):

        @BilinearForm
        def dudv(u, v, w):
            du, dv = u[1], v[1]
            return sum(du * dv)

        @BilinearForm
        def uv(u, v, w):
            return u * v

        def F(x, y, z):
            return 2 * x ** 2 + 2 * y ** 2 - 6 * x * y * z

        @LinearForm
        def fv(v, w):
            return F(*w.x) * v

        def G(x, y, z):
            eps = 1e-6

            def circa(a, b):
                return (a - b < eps) * (a - b > -eps)

            return (circa(x, 1) * (3 - 3 * y ** 2 + 2 * y * z ** 3) +
                    circa(x, 0) * (-y * z ** 3) +
                    circa(y, 1) * (1 + x - 3 * x ** 2 + 2 * x * z ** 3) +
                    circa(y, 0) * (1 + x - x * z ** 3) +
                    circa(z, 1) * (1 + x + 4 * x * y - x ** 2 * y ** 2) +
                    circa(z, 0) * (1 + x - x ** 2 * y ** 2))

        @LinearForm
        def gv(v, w):
            return G(*w.x) * v

        hs = np.array([])
        H1err = np.array([])
        L2err = np.array([])

        for itr in range(0, 3):
            m = self.case[0]().refined(self.preref + itr)

            ib = InteriorBasis(m, self.case[1]())
            fb = FacetBasis(m, self.case[1]())

            A = asm(dudv, ib)
            f = asm(fv, ib)

            B = asm(uv, fb)
            g = asm(gv, fb)

            u = solve(A + B, f + g)

            L2, H1 = self.compute_error(m, ib, u)
            hs = np.append(hs, m.param())
            L2err = np.append(L2err, L2)
            H1err = np.append(H1err, H1)

        pfit = np.polyfit(np.log10(hs),
                          np.log10(np.sqrt(L2err ** 2 + H1err ** 2)), 1)
        self.assertGreater(pfit[0], self.limits[0])
        self.assertLess(pfit[0], self.limits[1])
        self.assertLess(H1err[-1], 0.08)
        self.assertLess(L2err[-1], 0.005)

    def compute_error(self, m, basis, U):
        uh, duh, *_ = basis.interpolate(U)
        dx = basis.dx
        x = basis.global_coordinates().value

        def u(x):
            return 1 + x[0] - x[0] ** 2 * x[1] ** 2 + x[0] * x[1] * x[2] ** 3

        def ux(x):
            return 1 - 2 * x[0] * x[1] ** 2 + x[1] * x[2] ** 3

        def uy(x):
            return -2 * x[0] ** 2 * x[1] + x[0] * x[2] ** 3

        def uz(x):
            return 3 * x[0] * x[1] * x[2] ** 2

        L2 = np.sqrt(np.sum((uh - u(x)) ** 2 * dx))

        if x.shape[0] == 3:
            H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x)) ** 2 +
                                        (duh[1] - uy(x)) ** 2 +
                                        (duh[2] - uz(x)) ** 2) * dx, axis=1)))
        elif x.shape[0] == 2:
            H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x)) ** 2 +
                                        (duh[1] - uy(x)) ** 2) * dx, axis=1)))
        else:
            H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x)) ** 2) * dx, axis=1)))

        return L2, H1


class FacetConvergenceHex1(FacetConvergenceTetP2):

    case = (MeshHex, ElementHex1)
    limits = (0.9, 1.1)
    preref = 2


class FacetConvergenceHexS2(FacetConvergenceTetP2):

    case = (MeshHex, ElementHexS2)
    limits = (1.9, 2.2)
    preref = 1


class FacetConvergenceTetP1(FacetConvergenceTetP2):

    case = (MeshTet, ElementTetP1)
    limits = (0.9, 1.1)
    preref = 2


if __name__ == '__main__':
    unittest.main()
