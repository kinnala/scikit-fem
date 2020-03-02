import unittest
from skfem import *
from skfem.helpers import *


class ConvergenceMorley(unittest.TestCase):

    case = (MeshTri, ElementTriMorley)
    prerefs = 3
    limits = (1.9, 2.1)
    abs_limit = 8e-5

    def runTest(self):
        m = self.case[0]()
        m.refine(self.prerefs)

        hs = []
        L2s = []

        for itr in range(3):
            e = self.case[1]()
            ib = InteriorBasis(m, e)

            t = 1.
            E = 1.
            nu = 0.3
            D = E * t ** 3 / (12. * (1. - nu ** 2))

            @BilinearForm
            def bilinf(u, v, w):

                def C(T):
                    trT = T[0, 0] + T[1, 1]
                    return E / (1. + nu) * \
                        np.array([[T[0, 0] + nu / (1. - nu) * trT, T[0, 1]],
                                  [T[1, 0], T[1, 1] + nu / (1. - nu) * trT]])

                return t ** 3 / 12.0 * ddot(C(dd(u)), dd(v))

            def load(x):
                return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

            @LinearForm
            def linf(v, w):
                return load(w.x) * v

            K = asm(bilinf, ib)
            f = asm(linf, ib)

            x = solve(*condense(K, f, D=ib.get_dofs().all('u')))

            X = ib.interpolate(x)

            def exact(x):
                return 1. / (4. * D * np.pi ** 4) * load(x)

            @Functional
            def error(w):
                return (w.w - exact(w.x)) ** 2

            L2 = np.sqrt(error.assemble(ib, w=X))

            L2s.append(L2)
            hs.append(m.param())
            m.refine()

        hs = np.array(hs)
        L2s = np.array(L2s)
        pfit = np.polyfit(np.log10(hs), np.log10(L2s), 1)
        self.assertGreater(pfit[0], self.limits[0])
        self.assertLess(pfit[0], self.limits[1])
        self.assertLess(L2s[-1], self.abs_limit)


class ConvergenceArgyris(ConvergenceMorley):

    case = (MeshTri, ElementTriArgyris)
    preref = 0
    limits = (2.9, 3.1)
    abs_limit = 5e-7
