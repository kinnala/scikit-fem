from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

from skfem.mesh import MeshTri
from skfem.element import ElementTriArgyris, ElementTriP2
from skfem.assembly import Dofs, InteriorBasis


class TestDofsKeepSkipOperations(TestCase):

    def runTest(self):

        m = MeshTri()
        e = ElementTriArgyris()
        basis = InteriorBasis(m, e)
        all_dofs = basis.get_dofs()

        assert_allclose(all_dofs.keep(['u', 'u_n']).keep('u'),
                        all_dofs.keep('u'))

        assert_allclose(all_dofs.skip(['u_x', 'u_y', 'u_xx',
                                       'u_xy', 'u_yy', 'u_n']),
                        all_dofs.keep('u'))

        assert_allclose(all_dofs,
                        all_dofs.skip('does_not_exist'))

        assert_allclose(np.empty((0,), dtype=np.int64),
                        all_dofs.keep('does_not_exist'))


class TestDofsNodalSubsets(TestCase):

    def runTest(self):

        m = MeshTri()
        e = ElementTriArgyris()
        basis = InteriorBasis(m, e)
        all_dofs = basis.get_dofs()

        self.assertEqual(len(all_dofs.keep('u').nodal), 1)


class TestDofsMerge(TestCase):

    def runTest(self):

        m = MeshTri()
        basis = InteriorBasis(m, ElementTriP2())
        D1 = basis.get_dofs(lambda x: x[0] == 0)
        D2 = basis.get_dofs(lambda x: x[0] == 1)
        D3 = basis.get_dofs(lambda x: x[1] == 1)
        D4 = basis.get_dofs(lambda x: x[1] == 0)
        assert_allclose(D1 | D2 | D3 | D4,
                        basis.get_dofs())
        assert_allclose(D1 + D2 + D3 + D4,
                        basis.get_dofs())


class TestDofsSubset(TestCase):

    def runTest(self):

        m = MeshTri()
        basis = InteriorBasis(m, ElementTriP2())
        dofs = basis.get_dofs()

        self.assertEqual(len(dofs.nodal['u']), 4)
        self.assertEqual(len(dofs.facet['u']), 4)
