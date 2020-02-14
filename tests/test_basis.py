from unittest import TestCase

import numpy as np

from skfem import *


class TestCompositeSplitting(TestCase):

    def runTest(self):
        """Solve Stokes problem, try splitting and other small things."""

        m = MeshTri()
        m.refine(4)
        e = ElementVectorH1(ElementTriP2()) * ElementTriP1()

        m.define_boundary('up', lambda x: x[1] == 1.)
        m.define_boundary('rest', lambda x: x[1] != 1.)

        basis = InteriorBasis(m, e)

        @BilinearForm
        def bilinf(u, p, v, q, w):
            from skfem.helpers import grad, ddot, div
            return (ddot(grad(u), grad(v)) - div(u) * q - div(v) * p
                    - 1e-2 * p * q)

        S = asm(bilinf, basis)

        D = basis.boundary_dofs(skip=['u^2'])
        x = basis.zeros()
        x[D['up'].all('u^1^1')] = .1

        x = solve(*condense(S, basis.zeros(), x=x, D=D))

        (u, u_basis), (p, p_basis) = basis.split(x)

        self.assertEqual(len(u), m.p.shape[1] * 2 + m.facets.shape[1] * 2)
        self.assertEqual(len(p), m.p.shape[1])

        self.assertTrue(np.sum(p - x[basis.nodal_dofs[2]]) < 1e-8)

        y = basis.interpolate(x)
        self.assertTrue('w^1' in y)
        self.assertTrue('w^2' in y)

        self.assertTrue((basis.doflocs[:, D['up'].all()][1] == 1.).all())
