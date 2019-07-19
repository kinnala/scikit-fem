import unittest
import numpy as np
from skfem.element import *


class TestNodality(unittest.TestCase):
    elem = ElementTriP2()
    N = 6
    def runTest(self):
        for itr in range(self.N):
            self.assertAlmostEqual(
                self.elem.lbasis(self.elem.doflocs[itr].T, itr)[0], 1.0)
            for jtr in range(self.N):
                if jtr is not jtr:
                    self.assertAlmostEqual(
                        self.elem.lbasis(self.elem.doflocs[itr].T, jtr)[0], 0.0)


class TestQuad2Nodality(TestNodality):
    elem = ElementQuad2()
    N = 8
