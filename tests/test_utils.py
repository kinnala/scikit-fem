"""Unit tests for utils package."""
import unittest
import numpy as np

from skfem.assembly import asm, InteriorBasis
from skfem.element import ElementTriP1
from skfem.utils import L2_projection
from skfem.mesh import MeshTri


class InitializeScalarField(unittest.TestCase):
    def runTest(self):
        mesh = MeshTri()
        mesh.refine(5)
        basis = InteriorBasis(mesh, ElementTriP1())
        
        def fun(x, y):
            return x**2 + y**2
        
        x = L2_projection(fun, basis)
        y = fun(*mesh.p)

        normest = np.linalg.norm(x - y)

        self.assertTrue(normest < 0.011,
                        msg = "|x-y| = {}".format(normest))


if __name__ == '__main__':
    unittest.main()
