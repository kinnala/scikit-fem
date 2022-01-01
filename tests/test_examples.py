from unittest import TestCase, main

import numpy as np


class TestEx01(TestCase):

    def runTest(self):
        import docs.examples.ex01 as ex01
        self.assertAlmostEqual(np.max(ex01.x), 0.073657185490792)


class TestEx02(TestCase):

    def runTest(self):
        import docs.examples.ex02 as ex02
        self.assertAlmostEqual(np.max(ex02.x), 0.001217973811129439)


class TestEx03(TestCase):

    def runTest(self):
        import docs.examples.ex03 as ex03
        self.assertAlmostEqual(ex03.L[0], 0.00418289)


class TestEx04(TestCase):

    def runTest(self):
        import docs.examples.ex04 as ex04
        self.assertAlmostEqual(np.max(ex04.vonmises1), 65.3220252757752)
        self.assertAlmostEqual(np.max(ex04.vonmises2), 68.68010634444957)


class TestEx05(TestCase):

    def runTest(self):
        import docs.examples.ex05 as ex05
        self.assertAlmostEqual(np.max(ex05.x), 0.93570751751091152)


class TestEx06(TestCase):

    def runTest(self):
        import docs.examples.ex06 as ex06
        self.assertAlmostEqual(np.max(ex06.x), 0.073651530833125131)


class TestEx07(TestCase):

    def runTest(self):
        import docs.examples.ex07 as ex07
        self.assertAlmostEqual(np.max(ex07.x), 0.0737144219329924)


class TestEx08(TestCase):

    def runTest(self):
        import docs.examples.ex08 as ex08  # noqa
        # only run the initialization, nothing to test


class TestEx09(TestCase):

    def runTest(self):
        import docs.examples.ex09 as ex09
        self.assertAlmostEqual(np.max(ex09.x), 0.05528520791811886, places=6)


class TestEx10(TestCase):

    def runTest(self):
        import docs.examples.ex10 as ex10
        self.assertAlmostEqual(np.mean(ex10.x), 0.277931521728906)


class TestEx11(TestCase):

    def runTest(self):
        import docs.examples.ex11 as ex11
        u = ex11.u
        ib = ex11.ib
        # since the mesh is symmetric, the mean values should equal to zero
        self.assertAlmostEqual(np.mean(u[ib.nodal_dofs[2, :]]), 0.0)
        self.assertAlmostEqual(np.mean(u[ib.nodal_dofs[1, :]]), 0.0)


class TestEx12(TestCase):

    def runTest(self):
        import docs.examples.ex12 as ex
        self.assertAlmostEqual(ex.area, np.pi, delta=1e-2)
        self.assertAlmostEqual(ex.k, 1 / 8 / np.pi, delta=1e-5)
        self.assertAlmostEqual(ex.k1, 1 / 4 / np.pi, delta=1e-5)


class TestEx13(TestCase):

    def runTest(self):
        import docs.examples.ex13 as ex
        u = ex.u
        A = ex.A
        current = ex.current
        self.assertAlmostEqual(current['ground'],
                               -2 * np.log(2) / np.pi,
                               delta=1e-3)
        self.assertAlmostEqual(u @ A @ u,
                               2 * np.log(2) / np.pi,
                               delta=1e-3)


class TestEx14(TestCase):

    def runTest(self):
        import docs.examples.ex14
        u = docs.examples.ex14.u
        A = docs.examples.ex14.A
        self.assertAlmostEqual(u @ A @ u, 8 / 3, delta=1e-2)


class TestEx15(TestCase):

    def runTest(self):
        import docs.examples.ex15 as ex15
        self.assertTrue(np.max(ex15.x) - 0.1234567 < 1e-5)


class TestEx16(TestCase):

    def runTest(self):
        import docs.examples.ex16 as ex16
        self.assertTrue(np.linalg.norm(np.array([0, 2, 6, 12, 20, 30])
                                       - ex16.ks) < 0.4)
        self.assertTrue(ex16.ks[-1], 30.309720458315521)


class TestEx17(TestCase):

    def runTest(self):
        from docs.examples.ex17 import T0
        self.assertAlmostEqual(*T0.values(), 2)


class TestEx18(TestCase):
    def runTest(self):
        import docs.examples.ex18 as ex  # noqa

        self.assertAlmostEqual(
            (ex.basis["psi"].probes(np.zeros((ex.mesh.dim(), 1))) @ ex.psi)[0],
            1 / 64,
            3,
        )

        self.assertLess(
            np.linalg.norm(
                ex.basis["p"].probes(np.array([[-0.5, 0.0, 0.5], [0.5, 0.5, 0.5]]))
                @ ex.pressure
                - [-1 / 8, 0, +1 / 8]
            ),
            1e-3,
        )


class TestEx19(TestCase):

    def runTest(self):
        import docs.examples.ex19 as ex  # noqa

        t, u = next(ex.evolve(0.0, ex.u_init))
        self.assertAlmostEqual(*[(ex.probe @ s)[0] for s in [ex.exact(t), u]], 4)


class TestEx20(TestCase):

    def runTest(self):
        import docs.examples.ex20 as ex
        psi0 = ex.psi0
        self.assertAlmostEqual(psi0, 1 / 64, delta=1e-3)


class TestEx21(TestCase):

    def runTest(self):
        import docs.examples.ex21 as ex
        x = ex.x
        K = ex.K
        L = ex.L[0]
        self.assertAlmostEqual(L, 50194.51136114997, delta=1)
        self.assertAlmostEqual(L, x[:, 0].T @ K @ x[:, 0], 4)


class TestEx22(TestCase):

    def runTest(self):
        import docs.examples.ex22 as ex
        u = ex.u
        K = ex.K
        self.assertAlmostEqual(u.T @ K @ u, 0.2131280267335294)


class TestEx23(TestCase):

    def runTest(self):
        import docs.examples.ex23 as ex
        self.assertAlmostEqual(max(ex.lmbda_list), ex.turning_point,
                               delta=5e-5)


class TestEx24(TestCase):

    def runTest(self):
        import docs.examples.ex24 as ex24  # noqa

        self.assertAlmostEqual(min(ex24.vorticity), -0.05171085161096803)


class TestEx25(TestCase):

    def runTest(self):
        import docs.examples.ex25 as ex25
        mu = np.mean(ex25.t)
        self.assertAlmostEqual(mu, 0.4642600944590631, places=5)
        self.assertAlmostEqual(np.mean(ex25.t0), mu, places=2)


class TestEx26(TestCase):

    def runTest(self):
        from docs.examples.ex26 import T0
        self.assertAlmostEqual(*T0.values(), delta=2e-4)


class TestEx27(TestCase):

    def runTest(self):
        import docs.examples.ex27 as ex
        _, psi = ex.psi.popitem()
        self.assertAlmostEqual(min(psi), -0.027043, delta=1e-6)
        self.assertAlmostEqual(max(psi), 0.6668, delta=1e-5)


class TestEx28(TestCase):

    def runTest(self):
        from docs.examples.ex28 import exit_interface_temperature as t
        self.assertAlmostEqual(*t.values(), delta=2e-4)


class TestEx29(TestCase):

    def runTest(self):
        from docs.examples.ex29 import c
        wavespeed = tuple(
            np.array(sorted(wavespeed, key=np.imag, reverse=True))
            for wavespeed in c.values())
        self.assertLess(np.linalg.norm(wavespeed[1] - wavespeed[0], np.inf),
                        5e-3)


class TestEx30(TestCase):

    def runTest(self):
        from docs.examples.ex30 import psi0
        self.assertAlmostEqual(psi0, 0.162/128, delta=1e-6)


class TestEx31(TestCase):

    def runTest(self):
        from docs.examples.ex31 import L
        self.assertAlmostEqual(L[0], 22.597202568397734, delta=1e-6)


class TestEx32(TestCase):

    def runTest(self):
        from docs.examples.ex32 import l2error_p
        self.assertLess(l2error_p, 1e-5)


class TestEx33(TestCase):

    def runTest(self):
        from docs.examples.ex33 import x
        self.assertAlmostEqual(np.max(x), 0.12220233975847579, delta=1e-8)


class TestEx34(TestCase):

    def runTest(self):
        from docs.examples.ex34 import err
        self.assertAlmostEqual(err, 0., delta=1e-13)


class TestEx35(TestCase):

    def runTest(self):
        from docs.examples.ex35 import Z
        # exact value depends also on mesh generation,
        # over which we don't have control.
        # tolerance is low, but might still break if mesh is slightly different
        self.assertAlmostEqual(Z, 52.563390368494424, delta=1e-1)


class TestEx36(TestCase):

    def runTest(self):
        from docs.examples.ex36 import du, dp, volume_deformed, norm_res
        self.assertAlmostEqual(np.linalg.norm(du),
                               16.530715141106377,
                               delta=1e-5)
        self.assertAlmostEqual(dp[0], -0.5, delta=1.e-8)
        self.assertAlmostEqual(volume_deformed, 1., delta=1.e-4)
        self.assertAlmostEqual(norm_res, 0., delta=1.e-8)



class TestEx37(TestCase):

    def runTest(self):
        from docs.examples.ex37 import u
        self.assertAlmostEqual(np.max(u), 0.05594193697362236)


class TestEx38(TestCase):

    def runTest(self):
        from docs.examples.ex38 import l2error
        self.assertLess(l2error, 3e-3)


class TestEx39(TestCase):

    def runTest(self):
        import docs.examples.ex39 as ex  # noqa

        t, u = next(ex.evolve(0.0, ex.u_init))
        self.assertAlmostEqual(*[(ex.probe @ s)[0] for s in [ex.exact(t), u]], 5)


class TestEx40(TestCase):

    def runTest(self):
        import docs.examples.ex40 as ex

        self.assertAlmostEqual(ex.u1.max(), 0.0748, delta=1e-3)
        self.assertAlmostEqual(ex.u2.max(), 0.0748, delta=1e-3)
        self.assertAlmostEqual(ex.u1.min(), 0.0, delta=3e-3)
        self.assertAlmostEqual(ex.u2.min(), 0.0, delta=3e-3)


class TestEx41(TestCase):

    def runTest(self):
        import docs.examples.ex41 as ex

        self.assertAlmostEqual(ex.y.max(), 0.025183404207706196)
        self.assertAlmostEqual(ex.y.min(), 0.0)


class TestEx42(TestCase):

    def runTest(self):
        import docs.examples.ex42 as ex

        self.assertAlmostEqual(ex.x.max(), 0.0009824131638261542, delta=1e-5)


class TestEx43(TestCase):

    def runTest(self):
        import docs.examples.ex43 as ex

        self.assertAlmostEqual(ex.u.max(), 0.2466622622014594, delta=1e-8)
