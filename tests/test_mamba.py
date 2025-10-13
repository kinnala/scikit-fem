"""Tests run in conda/mamba env due to challenging binary deps."""

import pytest
from unittest import TestCase, main


@pytest.mark.mpi_skip
class TestEx52(TestCase):

    def runTest(self):
        import docs.examples.ex52 as ex52
        self.assertAlmostEqual(ex52.xmax, 0.999610970807614)



@pytest.mark.mpi
def test_ex_53():

    import docs.examples.ex53 as ex53
    assert ex53.xmax == 0.9996111271291415
