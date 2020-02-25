from unittest import TestCase

import numpy as np


class TestEx01(TestCase):
    def runTest(self):
        import docs.examples.ex01 as ex01
        self.assertAlmostEqual(np.max(ex01.x), 0.07344576657)


class TestEx02(TestCase):
    def runTest(self):
        import docs.examples.ex02 as ex02
        self.assertAlmostEqual(np.max(ex02.x), 0.001217973811129439)


class TestEx03(TestCase):
    def runTest(self):
        import docs.examples.ex03 as ex03
        self.assertAlmostEqual(ex03.L[0], 0.00418289)


class TestEx04(TestCase):
    def runTest(self):
        import docs.examples.ex04 as ex04
        self.assertAlmostEqual(np.max(ex04.vonmises1), 64.57919892367978)
        self.assertAlmostEqual(np.max(ex04.vonmises2), 67.91419753783893)


class TestEx05(TestCase):
    def runTest(self):
        import docs.examples.ex05 as ex05
        self.assertAlmostEqual(np.max(ex05.x), 0.93570751751091152)


class TestEx06(TestCase):
    def runTest(self):
        import docs.examples.ex06 as ex06
        self.assertAlmostEqual(np.max(ex06.x), 0.073651530833125131)


class TestEx07(TestCase):
    def runTest(self):
        import docs.examples.ex07 as ex07
        self.assertAlmostEqual(np.max(ex07.x), 0.07869083767545548)


class TestEx08(TestCase):
    def runTest(self):
        import docs.examples.ex08 as ex08
        # only run the initialization, nothing to test


class TestEx09(TestCase):
    def runTest(self):
        import docs.examples.ex09 as ex09
        self.assertAlmostEqual(np.max(ex09.x), 0.055596791644282988)


class TestEx10(TestCase):
    def runTest(self):
        import docs.examples.ex10 as ex10
        self.assertAlmostEqual(np.mean(ex10.x), 0.277931521728906)


class TestEx11(TestCase):
    def runTest(self):
        import docs.examples.ex11 as ex11
        u = ex11.u
        ib = ex11.ib
        # since the mesh is symmetric, the mean values should equal to zero
        self.assertAlmostEqual(np.mean(u[ib.nodal_dofs[2, :]]), 0.0)
        self.assertAlmostEqual(np.mean(u[ib.nodal_dofs[1, :]]), 0.0)


class TestEx12(TestCase):
    def runTest(self):
        import docs.examples.ex12 as ex
        self.assertAlmostEqual(ex.area, np.pi, delta=1e-2)
        self.assertAlmostEqual(ex.k, 1 / 8 / np.pi, delta=1e-5)
        self.assertAlmostEqual(ex.k1, 1 / 4 / np.pi, delta=1e-5)


class TestEx13(TestCase):
    def runTest(self):
        import docs.examples.ex13 as ex
        u = ex.u
        A = ex.A
        self.assertAlmostEqual(u @ A @ u, 2 * np.log(2) / np.pi, delta=1e-3)


class TestEx14(TestCase):
    def runTest(self):
        import docs.examples.ex14
        u = docs.examples.ex14.u
        A = docs.examples.ex14.A
        self.assertTrue(((u @ A @ u) - 8 / 3) < 0.01)


class TestEx15(TestCase):
    def runTest(self):
        import docs.examples.ex15 as ex15
        self.assertTrue(np.max(ex15.x) - 0.1234567 < 1e-5)


class TestEx16(TestCase):
    def runTest(self):
        import docs.examples.ex16 as ex16
        self.assertTrue(np.linalg.norm(np.array([0, 2, 6, 12, 20, 30])
                                       - ex16.ks) < 0.4)
        self.assertTrue(ex16.ks[-1], 30.309720458315521)


class TestEx17(TestCase):
    def runTest(self):
        import docs.examples.ex17 as ex
        # TODO improve


class TestEx18(TestCase):
    def runTest(self):
        import docs.examples.ex18 as ex
        # TODO improve


class TestEx19(TestCase):
    def runTest(self):
        import docs.examples.ex19 as ex
        # TODO improve


class TestEx20(TestCase):
    def runTest(self):
        import docs.examples.ex20 as ex
        psi0 = ex.psi0
        self.assertAlmostEqual(psi0, 1 / 64, delta=1e-3)


class TestEx21(TestCase):
    def runTest(self):
        import docs.examples.ex21 as ex
        x = ex.x
        K = ex.K
        L = ex.L[0]
        self.assertAlmostEqual(L, 40085.40062937357, 4)
        self.assertAlmostEqual(L, x[:, 0].T @ K @ x[:, 0], 4)


class TestEx22(TestCase):
    def runTest(self):
        import docs.examples.ex22 as ex
        u = ex.u
        K = ex.K
        self.assertAlmostEqual(u.T @ K @ u, 0.2131280267335294)


class TestEx23(TestCase):
   def runTest(self):
      import docs.examples.ex23 as ex
      self.assertAlmostEqual(max(ex.lmbda_list), ex.turning_point,
                             delta=5e-5)


class TestEx24(TestCase):
    def runTest(self):
        import docs.examples.ex24 as ex24


class TestEx25(TestCase):
    def runTest(self):
        import docs.examples.ex25 as ex25
        mu = np.mean(ex25.t)
        self.assertAlmostEqual(mu, 0.4642600944590631, places=5)
        self.assertAlmostEqual(np.mean(ex25.t0), mu, places=2)


class TestEx26(TestCase):
    def runTest(self):
        import docs.examples.ex26 as ex26


class TestEx27(TestCase):
   def runTest(self):
      import docs.examples.ex27 as ex
      _, psi = ex.psi.popitem()
      self.assertAlmostEqual(min(psi), -0.02704307069920056)
      self.assertAlmostEqual(max(psi), 0.6668969079018402)


class TestEx28(TestCase):
    def runTest(self):
        from docs.examples.ex28 import exit_interface_temperature as t
        self.assertAlmostEqual(*t.values(), delta=2e-4)


class TestEx29(TestCase):
    def runTest(self):
        from docs.examples.ex29 import c
        wavespeed = tuple(
            np.array(sorted(wavespeed, key=np.imag, reverse=True))
            for wavespeed in c.values())
        self.assertLess(np.linalg.norm(wavespeed[1] - wavespeed[0], np.inf),
                        5e-3)


class TestEx30(TestCase):
    def runTest(self):
        from docs.examples.ex30 import psi0
        self.assertAlmostEqual(psi0, 0.162/128, delta=1e-6)


if __name__ == '__main__':
    main()
