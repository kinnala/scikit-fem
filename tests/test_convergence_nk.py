from unittest import TestCase
import numpy as np

from skfem import *
from skfem.helpers import dot, curl, grad


# Convergence test based on ex46
class ConvergenceNedelec(TestCase):
    case = (MeshTri, lambda: ElementTriN1() * ElementTriP1())
    intorder=3
    limits = (1.9, 2.1)
    abs_limit = 1e-3
    N_start = 6
    width = 1.0
    height = 0.5
    
    def runTest(self):
        
        m = self.case[0].init_tensor(
            np.linspace(0, self.width, self.N_start),
            np.linspace(0, self.height, max(2, int(self.N_start * self.height / self.width)))
        )

        hs = []
        errs = []

        one_over_u_r = 1.
        exact_eig = np.pi ** 2

        for itr in range(5):
            e = self.case[1]()
            basis = Basis(m, e,intorder=self.intorder)
            epsilon = lambda x: 1. + 0. * x[0]

            @BilinearForm
            def aform(E, lam, v, mu, w):
                return one_over_u_r * curl(E) * curl(v)

            @BilinearForm
            def gauge(E, lam, v, mu, w):
                return dot(grad(lam), v) + dot(E, grad(mu))

            @BilinearForm
            def bform(E, lam, v, mu, w):
                return epsilon(w.x) * dot(E, v)

            A = aform.assemble(basis)
            B = bform.assemble(basis)
            C = gauge.assemble(basis)

            lams, xs = solve(*condense(A + C, B, D=basis.get_dofs()),
                             solver=solver_eigen_scipy_sym(k=3, sigma=0.1))

            error = np.min(np.abs(lams - exact_eig))

            errs.append(error)
            hs.append(m.param())
            m = m.refined()

        hs = np.array(hs)
        errs = np.array(errs)
        
        pfit = np.polyfit(np.log10(hs), np.log10(errs), 1)
        slope = pfit[0]
        #print(slope)
        self.assertGreater(slope, self.limits[0])
        self.assertLess(slope, self.limits[1])
        self.assertLess(errs[-1], self.abs_limit)


class ConvergenceNedelecN2(ConvergenceNedelec):
    case = (MeshTri, lambda: ElementTriN2() * ElementTriP2())
    limits = (3.9, 4.1)
    intorder = 5
    N_start = 4

class ConvergenceNedelecN3(ConvergenceNedelec):
    case = (MeshTri, lambda: ElementTriN3() * ElementTriP3())
    limits = (5.8, 6.2) # Higher order is much more sensitive. The slope is 5.907 for these values,
    # but it can move a bit further for different meshes/integration orders.
    intorder = 9
    N_start = 3


if __name__ == "__main__":
    import unittest
    unittest.main()