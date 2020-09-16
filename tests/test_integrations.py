import numpy as np

import pytest

pytest.importorskip("mumps")


def test_pymumps():
    import docs.integrations.solver_pymumps as example
    np.testing.assert_almost_equal(np.linalg.norm(example.x, np.inf), 0.05528520791811886)
