"""These tests run the examples and check that their output stays constant."""

import unittest

import numpy as np


class TestEx01(unittest.TestCase):
    """Run examples/ex01.py"""

    def runTest(self):
        import docs.examples.ex01 as ex01
        self.assertAlmostEqual(np.max(ex01.x), 0.07344576657)


class TestEx02(unittest.TestCase):
    """Run examples/ex02.py"""

    def runTest(self):
        import docs.examples.ex02 as ex02
        self.assertAlmostEqual(np.max(ex02.x), 0.001217973811129439)


class TestEx03(unittest.TestCase):
    """Run examples/ex03.py"""

    def runTest(self):
        import docs.examples.ex03 as ex03
        self.assertAlmostEqual(ex03.L[0], 0.00418289)


class TestEx04(unittest.TestCase):
    """Run examples/ex04.py"""

    def runTest(self):
        import docs.examples.ex04 as ex04
        self.assertAlmostEqual(np.max(ex04.vonmises1), 64.57919892367978)
        self.assertAlmostEqual(np.max(ex04.vonmises2), 67.91419753783893)


class TestEx05(unittest.TestCase):
    """Run examples/ex05.py"""

    def runTest(self):
        import docs.examples.ex05 as ex05
        self.assertAlmostEqual(np.max(ex05.x), 0.93570751751091152)


class TestEx06(unittest.TestCase):
    """Run examples/ex06.py"""

    def runTest(self):
        import docs.examples.ex06 as ex06
        self.assertAlmostEqual(np.max(ex06.x), 0.073651530833125131)


class TestEx07(unittest.TestCase):
    """Run examples/ex07.py"""

    def runTest(self):
        import docs.examples.ex07 as ex07
        self.assertAlmostEqual(np.max(ex07.x), 0.07869083767545548)


class TestEx08(unittest.TestCase):
    """Run examples/ex08.py"""

    def runTest(self):
        pass
        # only run the initialization, nothing to test


class TestEx09(unittest.TestCase):
    """Run examples/ex09.py"""

    def runTest(self):
        import docs.examples.ex09 as ex09
        self.assertAlmostEqual(np.max(ex09.x), 0.055596791644282988)


class TestEx10(unittest.TestCase):
    """Run examples/ex10.py"""

    def runTest(self):
        import docs.examples.ex10 as ex10
        self.assertAlmostEqual(np.mean(ex10.x), 0.277931521728906)


class TestEx11(unittest.TestCase):
    """Run examples/ex11.py"""

    def runTest(self):
        import docs.examples.ex11 as ex11
        u = ex11.u
        ib = ex11.ib
        # since the mesh is symmetric, the mean values should equal to zero
        self.assertAlmostEqual(np.mean(u[ib.nodal_dofs[2, :]]), 0.0)
        self.assertAlmostEqual(np.mean(u[ib.nodal_dofs[1, :]]), 0.0)


class TestEx12(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex12 as ex
        self.assertAlmostEqual(ex.area, np.pi, delta=1e-2)
        self.assertAlmostEqual(ex.k, 1 / 8 / np.pi, delta=1e-5)
        self.assertAlmostEqual(ex.k1, 1 / 4 / np.pi, delta=1e-5)


class TestEx13(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex13 as ex
        u = ex.u
        A = ex.A
        self.assertAlmostEqual(u @ A @ u, 2 * np.log(2) / np.pi, delta=1e-3)


class TestEx14(unittest.TestCase):
    """Run examples/ex14.py"""

    def runTest(self):
        import docs.examples.ex14
        u = docs.examples.ex14.u
        A = docs.examples.ex14.A
        self.assertTrue(((u @ A @ u) - 8 / 3) < 0.01)


class TestEx15(unittest.TestCase):
    """Run examples/ex15.py"""

    def runTest(self):
        import docs.examples.ex15
        self.assertTrue(np.max(docs.examples.ex15.x) - 0.1234567 < 1e-5)


class TestEx16(unittest.TestCase):
    """Run examples/ex16.py"""

    def runTest(self):
        import docs.examples.ex16
        self.assertTrue(np.linalg.norm(np.array([0, 2, 6, 12, 20, 30]) - docs.examples.ex16.ks) < 0.4)
        self.assertTrue(docs.examples.ex16.ks[-1], 30.309720458315521)


class TestEx17(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex17 as ex


class TestEx18(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex18 as ex


class TestEx19(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex19 as ex


class TestEx20(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex20 as ex
        psi0 = ex.psi0
        self.assertAlmostEqual(psi0, 1 / 64, delta=1e-3)


class TestEx21(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex21 as ex
        y = ex.y
        K = ex.K
        L = ex.L[0]
        self.assertAlmostEqual(y.T @ K @ y, 253739.1255284172, 4)
        self.assertAlmostEqual(L, 253739.12552853522, 4)


class TestEx22(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex22 as ex
        u = ex.u
        K = ex.K
        self.assertAlmostEqual(u.T @ K @ u, 0.2131280267335294)


class TestEx23(unittest.TestCase):
   def runTest(self):
      import docs.examples.ex23 as ex
      self.assertAlmostEqual(max(ex.lmbda_list), ex.turning_point,
                             delta=5e-5)

class TestEx24(unittest.TestCase):
    def runTest(self):
        pass


class TestEx25(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex25 as ex
        mu = np.mean(ex.t)
        self.assertAlmostEqual(mu, 0.4642600944590631, places=5)
        self.assertAlmostEqual(np.mean(ex.t0), mu, places=2)


class TestEx26(unittest.TestCase):
    def runTest(self):
        pass


class TestEx27(unittest.TestCase):
   def runTest(self):
      import docs.examples.ex27 as ex
      _, psi = ex.psi.popitem()
      self.assertAlmostEqual(min(psi), -0.027510982599400716)
      self.assertAlmostEqual(max(psi), 0.6670038654518545)


class TestEx28(unittest.TestCase):
    def runTest(self):
        from docs.examples.ex28 import exit_interface_temperature as t
        self.assertAlmostEqual(*t.values(), delta=2e-4)


class TestEx29(unittest.TestCase):
    def runTest(self):
        from docs.examples.ex29 import c
        wavespeed = tuple(
            np.array(sorted(wavespeed, key=np.imag, reverse=True))
            for wavespeed in c.values())
        self.assertLess(np.linalg.norm(wavespeed[1] - wavespeed[0], np.inf),
                        5e-3)


class TestEx30(unittest.TestCase):
    def runTest(self):
        from docs.examples.ex30 import psi0
        self.assertAlmostEqual(psi0, 0.162/128, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
