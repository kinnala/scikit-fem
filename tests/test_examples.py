import unittest
import numpy as np
from skfem import *

"""
These tests run the examples and check that their output stays constant.
"""

class TestEx1(unittest.TestCase):
    """Run examples/ex01.py"""
    def runTest(self):
        import examples.ex01
        self.assertAlmostEqual(np.max(examples.ex01.x), 0.07344576657)

class TestEx3(unittest.TestCase):
    """Run examples/ex03.py"""
    def runTest(self):
        import examples.ex03
        self.assertAlmostEqual(examples.ex03.L[0], 0.00418289)

class TestEx5(unittest.TestCase):
    """Run examples/ex05.py"""
    def runTest(self):
        import examples.ex05
        self.assertAlmostEqual(np.max(examples.ex05.x), 0.93570751751091152)

class TestEx6(unittest.TestCase):
    """Run examples/ex06.py"""
    def runTest(self):
        import examples.ex06
        self.assertAlmostEqual(np.max(examples.ex06.x), 0.073651530833125131)

class TestEx7(unittest.TestCase):
    """Run examples/ex07.py"""
    def runTest(self):
        import examples.ex07
        self.assertAlmostEqual(np.max(examples.ex07.x), 0.077891428529719878)

class TestEx9(unittest.TestCase):
    """Run examples/ex09.py"""
    def runTest(self):
        import examples.ex09
        self.assertAlmostEqual(np.max(examples.ex09.x), 0.054313059564406921)

