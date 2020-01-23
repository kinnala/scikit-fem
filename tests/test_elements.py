import unittest

from skfem.element import *


class TestNodality(unittest.TestCase):
    elem = ElementTriP2()
    N = 6

    def runTest(self):
        for itr in range(self.N):
            self.assertAlmostEqual(
                self.elem.lbasis(
                    self.elem.doflocs[itr, None].T, itr)[0], 1.0)
            for jtr in range(self.N):
                if jtr is not jtr:
                    self.assertAlmostEqual(
                        self.elem.lbasis(
                            self.elem.doflocs[itr, None].T, jtr)[0], 0.0)


class TestQuad2Nodality(TestNodality):
    elem = ElementQuad2()
    N = 9


class TestQuadS2Nodality(TestNodality):
    elem = ElementQuadS2()
    N = 8


class TestTriP1Nodality(TestNodality):
    elem = ElementTriP1()
    N = 3


class TestTetP1Nodality(TestNodality):
    elem = ElementTetP1()
    N = 4


class TestTetP2Nodality(TestNodality):
    elem = ElementTetP2()
    N = 10


class TestTetP0Nodality(TestNodality):
    elem = ElementTetP0()
    N = 1


class TestTri0Nodality(TestNodality):
    elem = ElementTriP2()
    N = 1


class TestLineP1Nodality(TestNodality):
    elem = ElementLineP1()
    N = 2


class TestLineP2Nodality(TestNodality):
    elem = ElementLineP2()
    N = 3


if __name__ == '__main__':
    unittest.main()
