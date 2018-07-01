import unittest
import numpy as np
from skfem import *

class TestEx1(unittest.TestCase):
    """Run examples/ex1.py"""
    def runTest(self):
        import examples.ex1
        self.assertAlmostEqual(np.max(examples.ex1.x), 0.07344576657)

