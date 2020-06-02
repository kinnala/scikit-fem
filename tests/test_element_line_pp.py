from unittest import TestCase
from skfem.element.element_line.element_line_pp import ElementLinePp


class TestElementLinePp(TestCase):

    def test_p_less_than_1_error(self):
        """ Tests that exception is thrown when trying to initialize element
        with p < 1. """
        with self.assertRaises(ValueError):
            ElementLinePp(0)

