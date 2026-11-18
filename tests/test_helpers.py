import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfem.helpers import inv


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize('batched', [False, True])
def test_inverse_input_dtypes(dim, dtype, batched):
    A = (3 * np.eye(dim) + np.ones((dim, dim))).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        A *= 1. + .25j
    if batched:
        scales = np.arange(1, 7, dtype=dtype).reshape(2, 3)
        A = A[:, :, None, None] * scales
    original = A.copy()
    expected = np.moveaxis(np.linalg.inv(np.moveaxis(A, (0, 1), (-2, -1))),
                           (-2, -1), (0, 1))

    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = inv(A)

    assert result.dtype == expected.dtype
    assert_allclose(result, expected, rtol=2e-5, atol=2e-7)
    np.testing.assert_array_equal(A, original)


@pytest.mark.parametrize('dim', [2, 3])
def test_inverse_promotes_before_integer_arithmetic(dim):
    A = (1000 * (3 * np.eye(dim) + np.ones((dim, dim)))).astype(np.int16)
    original = A.copy()
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = inv(A)
    assert_allclose(result, np.linalg.inv(A))
    np.testing.assert_array_equal(A, original)
