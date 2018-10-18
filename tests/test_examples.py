import unittest
import numpy as np
from skfem import *

"""
These tests run the examples and check that their output stays constant.
"""

class TestEx01(unittest.TestCase):
    """Run examples/ex01.py"""
    def runTest(self):
        import examples.ex01
        self.assertAlmostEqual(np.max(examples.ex01.x), 0.07344576657)

class TestEx02(unittest.TestCase):
    """Run examples/ex02.py"""
    def runTest(self):
        import examples.ex02
        self.assertAlmostEqual(np.max(examples.ex02.x), 0.001217973811129439)

class TestEx03(unittest.TestCase):
    """Run examples/ex03.py"""
    def runTest(self):
        import examples.ex03
        self.assertAlmostEqual(examples.ex03.L[0], 0.00418289)

class TestEx04(unittest.TestCase):
    """Run examples/ex04.py"""
    def runTest(self):
        import examples.ex04
        self.assertAlmostEqual(np.max(examples.ex04.x), 0.0294118495011)

class TestEx05(unittest.TestCase):
    """Run examples/ex05.py"""
    def runTest(self):
        import examples.ex05
        self.assertAlmostEqual(np.max(examples.ex05.x), 0.93570751751091152)

class TestEx06(unittest.TestCase):
    """Run examples/ex06.py"""
    def runTest(self):
        import examples.ex06
        self.assertAlmostEqual(np.max(examples.ex06.x), 0.073651530833125131)

class TestEx07(unittest.TestCase):
    """Run examples/ex07.py"""
    def runTest(self):
        import examples.ex07
        self.assertAlmostEqual(np.max(examples.ex07.x), 0.077891428529719878)

class TestEx08(unittest.TestCase):
    """Run examples/ex08.py"""
    def runTest(self):
        import examples.ex08
        # only run the initialization, nothing to test

class TestEx09(unittest.TestCase):
    """Run examples/ex09.py"""
    def runTest(self):
        import examples.ex09
        self.assertAlmostEqual(np.max(examples.ex09.x), 0.055596791644282988)

class TestEx10(unittest.TestCase):
    """Run examples/ex10.py"""
    def runTest(self):
        import examples.ex10
        self.assertAlmostEqual(np.mean(examples.ex10.x), 0.14566021189334058)

class TestEx11(unittest.TestCase):
    """Run examples/ex11.py"""
    def runTest(self):
        import examples.ex11
        u = examples.ex11.u
        ib = examples.ex11.ib
        # since the mesh is symmetric, the mean values should equal to zero
        self.assertAlmostEqual(np.mean(u[ib.nodal_dofs[2, :]]), 0.0)
        self.assertAlmostEqual(np.mean(u[ib.nodal_dofs[1, :]]), 0.0)

class TestEx14(unittest.TestCase):
    """Run examples/ex14.py"""
    def runTest(self):
        import examples.ex14
        u = examples.ex14.u
        A = examples.ex14.A
        self.assertTrue(((u @ A @ u) - 8/3) < 0.01)

class TestEx15(unittest.TestCase):
    """Run examples/ex15.py"""
    def runTest(self):
        import examples.ex15
        self.assertTrue(np.max(examples.ex15.x) - 0.1234567 < 1e-5)

class TestEx16(unittest.TestCase):
    """Run examples/ex16.py"""
    def runTest(self):
        import examples.ex16
        self.assertTrue(np.linalg.norm(np.array([0,2,6,12,20,30])-examples.ex16.ks) < 0.4)
        self.assertTrue(examples.ex16.ks[-1], 30.309720458315521)
