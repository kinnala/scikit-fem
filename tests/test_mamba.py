"""Tests run in conda/mamba env due to challenging binary deps."""

from unittest import TestCase, main


class TestEx52(TestCase):

    def runTest(self):
        import docs.examples.ex52 as ex52
        self.assertAlmostEqual(ex52.xmax, 0.999610970807614)
