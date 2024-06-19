from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose
import pytest

from skfem.mesh import MeshTri
from skfem.element import ElementTriArgyris, ElementTriP2
from skfem.assembly import Dofs, CellBasis


class TestDofsKeepSkipOperations(TestCase):

    def runTest(self):

        m = MeshTri()
        e = ElementTriArgyris()
        basis = CellBasis(m, e)
        all_dofs = basis.get_dofs()

        assert_allclose(all_dofs.keep(['u', 'u_n']).keep('u'),
                        all_dofs.keep('u'))

        assert_allclose(all_dofs.drop(['u_x', 'u_y', 'u_xx',
                                       'u_xy', 'u_yy', 'u_n']),
                        all_dofs.keep('u'))

        assert_allclose(all_dofs,
                        all_dofs.drop('does_not_exist'))

        assert_allclose(np.empty((0,), dtype=np.int32),
                        all_dofs.keep('does_not_exist'))


class TestDofsNodalSubsets(TestCase):

    def runTest(self):

        m = MeshTri()
        e = ElementTriArgyris()
        basis = CellBasis(m, e)
        all_dofs = basis.get_dofs()

        self.assertEqual(len(all_dofs.keep('u').nodal), 1)
        self.assertTrue('u' in all_dofs.keep('u').nodal)

        self.assertEqual(len(all_dofs.keep('u_n').facet), 1)
        self.assertEqual(len(all_dofs.drop('u').facet), 1)

        all_dofs = basis.dofs.get_facet_dofs(
            m.facets_satisfying(lambda x: 1 + x[0] * 0))

        self.assertEqual(len(all_dofs.keep('u_n').facet), 1)
        self.assertEqual(len(all_dofs.drop('u').facet), 1)


class TestDofsMerge(TestCase):

    def runTest(self):

        m = MeshTri()
        basis = CellBasis(m, ElementTriP2())
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
        basis = CellBasis(m, ElementTriP2())
        dofs = basis.get_dofs()

        self.assertEqual(len(dofs.nodal['u']), 4)
        self.assertEqual(len(dofs.facet['u']), 4)


class TestDofsFromBoundaries(TestCase):

    def runTest(self):

        m = MeshTri().with_boundaries({
            'left': lambda x: x[0] == 0,
            'right': lambda x: x[0] == 1,
            'top': lambda x: x[1] == 1,
            'bottom': lambda x: x[1] == 0,
        })
        basis = CellBasis(m, ElementTriArgyris())
        assert_allclose(
            basis.get_dofs({'left', 'right'}).all(),
            np.sort(
                np.hstack([basis.get_dofs('left').all(), basis.get_dofs('right').all()])
            ),
        )
        assert_allclose(
            basis.get_dofs(('left', 'right', 'top', 'bottom')).all(),
            basis.get_dofs().all(),
        )


def test_nonexisting_keys():

    m = MeshTri()
    basis = CellBasis(m, ElementTriP2())

    with pytest.raises(ValueError):
        basis.get_dofs("does not exist")

    with pytest.raises(ValueError):
        basis.get_dofs(elements="does not exist")
