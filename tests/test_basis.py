from unittest import TestCase

import numpy as np

from skfem import *


class TestCompositeSplitting(TestCase):

    def runTest(self):
        """Solve Stokes problem, try splitting and other small things."""

        m = MeshTri()
        m.refine()
        m.define_boundary('centreline', lambda x: x[0] == .5,
                          boundaries_only=False)
        m.refine(3)

        e = ElementVectorH1(ElementTriP2()) * ElementTriP1()

        m.define_boundary('up', lambda x: x[1] == 1.)
        m.define_boundary('rest', lambda x: x[1] != 1.)

        basis = InteriorBasis(m, e)
        self.assertEqual(
            basis.get_dofs(m.boundaries['centreline']).all().size,
            (2 + 1) * (2**(1 + 3) + 1) + 2 * 2**(1 + 3))
        self.assertEqual(basis.boundary_dofs()['centreline'].all().size,
                         (2 + 1) * (2**(1 + 3) + 1) + 2 * 2**(1 + 3))

        
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


class TestDoflocs(TestCase):

    mesh = MeshTri
    elems = [
        ElementTriP1(),
        ElementTriRT0(),
        ElementTriP2(),
        ElementTriArgyris(),
    ]

    def runTest(self):
        """Test that global and local doflocs match in ref. domain."""

        for e in self.elems:
            self.assertEqual(
                np.sum(InteriorBasis(self.mesh.init_refdom(), e).doflocs
                       - e.doflocs.T),
                0.0
            )


class TestDoflocsQuad(TestDoflocs):

    mesh = MeshQuad
    elems = [
        ElementQuadP(4),
        ElementQuad2(),
    ]


class TestDoflocsTet(TestDoflocs):

    mesh = MeshTet
    elems = [
        ElementTetP2(),
        ElementTetRT0(),
        ElementTetN0(),
    ]


class TestDoflocsHex(TestDoflocs):

    mesh = MeshHex
    elems = [
        ElementHex1(),
    ]
