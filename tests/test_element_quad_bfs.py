from unittest import TestCase
from skfem.element.element_quad.element_quad_bfs import ElementQuadBFS


class TestElementQuadBFS(TestCase):

    def test_throw_index_error(self):
        """ Tests that exception is thrown when i % 4 not in (0, 1, 2, 3)."""
        element = ElementQuadBFS()
        with self.assertRaises(ValueError):
            element.gdof(0, 0, 0.5)
