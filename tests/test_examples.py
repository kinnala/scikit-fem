"""These tests run the examples and check that their output stays constant."""
import unittest
import numpy as np
from skfem import *


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
        import docs.examples.ex08 as ex08
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


class TestEx14(unittest.TestCase):
    """Run examples/ex14.py"""
    def runTest(self):
        import docs.examples.ex14
        u = docs.examples.ex14.u
        A = docs.examples.ex14.A
        self.assertTrue(((u @ A @ u) - 8/3) < 0.01)


class TestEx15(unittest.TestCase):
    """Run examples/ex15.py"""
    def runTest(self):
        import docs.examples.ex15
        self.assertTrue(np.max(docs.examples.ex15.x) - 0.1234567 < 1e-5)


class TestEx16(unittest.TestCase):
    """Run examples/ex16.py"""
    def runTest(self):
        import docs.examples.ex16
        self.assertTrue(np.linalg.norm(np.array([0,2,6,12,20,30])-docs.examples.ex16.ks) < 0.4)
        self.assertTrue(docs.examples.ex16.ks[-1], 30.309720458315521)


class TestEx12(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex12 as ex
        area = ex.area
        self.assertTrue((area - np.pi)<1e-3)


class TestEx13(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex13 as ex
        u = ex.u
        A = ex.A
        self.assertAlmostEqual(u @ A @ u, 2 * np.log(2) / np.pi, delta=1e-3)


class TestEx20(unittest.TestCase):
    def runTest(self):
        import docs.examples.ex20 as ex
        psi0 = ex.psi0
        self.assertTrue((psi0 - 1/64)<1e-3)


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
