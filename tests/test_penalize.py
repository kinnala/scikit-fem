import numpy as np
import pytest
import scipy.sparse

from skfem import penalize, solve


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('matrix_dtype', [np.float64, np.complex128])
@pytest.mark.parametrize(
    'rhs_dtype', [np.int32, np.float32, np.float64, np.complex128])
@pytest.mark.parametrize('selector', ['D', 'I'])
def test_penalize_complex_values(matrix_dtype, rhs_dtype, selector):
    A = scipy.sparse.diags([-1., 2., -1.], [-1, 0, 1],
                           shape=(3, 3), format='csr', dtype=matrix_dtype)
    b = np.array([0, 1, 0], dtype=rhs_dtype)
    x = np.array([1. + 2.j, 0., 3. + 4.j])
    original_A, original_b, original_x = A.copy(), b.copy(), x.copy()
    dofs = {'D': np.array([0, 2])} if selector == 'D' else {'I': np.array([1])}

    Aout, bout = penalize(A, b, x=x, epsilon=1e-4, **dofs)

    reference_A = np.array([[1e4, -1., 0.], [-1., 2., -1.],
                            [0., -1., 1e4]])
    reference_b = np.array([1e4 + 2e4j, 1., 3e4 + 4e4j])
    np.testing.assert_allclose(solve(Aout, bout),
                               np.linalg.solve(reference_A, reference_b))
    assert bout.dtype == np.complex128
    np.testing.assert_array_equal(A.toarray(), original_A.toarray())
    np.testing.assert_array_equal(b, original_b)
    np.testing.assert_array_equal(x, original_x)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('overwrite', [False, True])
def test_penalize_complex_rhs_control(dtype, overwrite):
    A = scipy.sparse.diags([-1., 2., -1.], [-1, 0, 1],
                           shape=(3, 3), format='csr')
    b = np.zeros(3, dtype=dtype)
    x = np.array([1. + 2.j, 0., 3. + 4.j], dtype=dtype)
    Aout, bout = penalize(
        A, b, x=x, D=np.array([0, 2]), epsilon=1e-4, overwrite=overwrite)
    reference = np.linalg.solve([[1e4, -1., 0.], [-1., 2., -1.],
                                 [0., -1., 1e4]],
                                [1e4 + 2e4j, 0., 3e4 + 4e4j])
    np.testing.assert_allclose(solve(Aout, bout), reference, rtol=1e-6)
    assert (Aout is A) == overwrite
    assert (bout is b) == overwrite


@pytest.mark.parametrize('matrix_dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('rhs_dtype', [np.float32, np.float64])
def test_penalize_complex_overwrite_error(matrix_dtype, rhs_dtype):
    A = scipy.sparse.diags([-1., 2., -1.], [-1, 0, 1],
                           shape=(3, 3), format='csr', dtype=matrix_dtype)
    b = np.zeros(3, dtype=rhs_dtype)
    x = np.array([1. + 2.j, 0., 3. + 4.j])
    original_A, original_b, original_x = A.copy(), b.copy(), x.copy()
    with pytest.raises(ValueError, match='complex right hand side'):
        penalize(A, b, x=x, D=np.array([0, 2]), epsilon=1e-4, overwrite=True)
    np.testing.assert_array_equal(A.toarray(), original_A.toarray())
    np.testing.assert_array_equal(b, original_b)
    np.testing.assert_array_equal(x, original_x)


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('rhs_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('values', ['zero', 'real_boundary', 'default'])
@pytest.mark.parametrize('overwrite', [False, True])
def test_penalize_real_values_from_complex_array(rhs_dtype, values, overwrite):
    A = scipy.sparse.diags([-1., 2., -1.], [-1, 0, 1],
                           shape=(3, 3), format='csr', dtype=complex)
    b = np.array([0., 1., 0.], dtype=rhs_dtype)
    x = {'zero': np.zeros(3, dtype=complex),
         'real_boundary': np.array([1., 3.j, 3.]), 'default': None}[values]
    expected_b = [1e4, 1., 3e4] if values == 'real_boundary' else [0., 1., 0.]
    Aout, bout = penalize(
        A, b, x=x, D=np.array([0, 2]), epsilon=1e-4, overwrite=overwrite)
    np.testing.assert_array_equal(bout, expected_b)
    assert (Aout is A) == overwrite
    assert (bout is b) == overwrite
    assert not np.iscomplexobj(bout)
    if overwrite:
        assert bout.dtype == rhs_dtype


def test_penalize_integer_rhs_copy():
    A = scipy.sparse.eye(3, format='csr')
    b = np.array([0, 1, 0])
    x = np.array([1, 0, 2])
    Aout, bout = penalize(A, b, x=x, D=np.array([0, 2]), epsilon=.3)
    np.testing.assert_allclose(solve(Aout, bout), [1., 1., 2.])
    assert np.issubdtype(bout.dtype, np.floating)
    np.testing.assert_array_equal(b, [0, 1, 0])
