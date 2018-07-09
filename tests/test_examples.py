import unittest
import numpy as np
from skfem import *

"""
These tests run the examples and check that their output stays constant.
"""

class TestEx1(unittest.TestCase):
    """Run examples/ex1.py"""
    def runTest(self):
        import examples.ex1
        self.assertAlmostEqual(np.max(examples.ex1.x), 0.07344576657)

class TestEx3(unittest.TestCase):
    """Run examples/ex3.py"""
    def runTest(self):
        import examples.ex3
        self.assertAlmostEqual(examples.ex3.L[0], 0.00418289)

class TestEx5(unittest.TestCase):
    """Run examples/ex5.py"""
    def runTest(self):
        import examples.ex5
        self.assertAlmostEqual(np.max(examples.ex5.x), 0.93570751751091152)

class TestEx6(unittest.TestCase):
    """Run examples/ex6.py"""
    def runTest(self):
        import examples.ex6
        self.assertAlmostEqual(np.max(examples.ex6.x), 0.073651530833125131)

class TestEx7(unittest.TestCase):
    """Run examples/ex7.py"""
    def runTest(self):
        import examples.ex7
        self.assertAlmostEqual(np.max(examples.ex7.x), 0.077891428529719878)

class TestEx16(unittest.TestCase):
    """Run examples/ex16.py"""
    def runTest(self):
        import examples.ex16
        self.assertAlmostEqual(np.max(examples.ex16.x), 0.0558240124419)
