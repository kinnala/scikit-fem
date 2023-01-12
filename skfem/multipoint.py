from scipy.sparse import csr_array
import numpy as np


def multipoint(D, *Ixs, g=None):
    if g is None:
        g = np.zeros(D.N)
    shape = D.N, D.N-len(D.flatten())

    I_rows = np.arange(D.N-len(D.flatten()))
    I_columns = np.setdiff1d(np.arange(D.N), D)

    result = csr_array(
        (np.ones((D.N-len(D.flatten()),)), (I_columns, I_rows)), shape=shape
    )

    for Ix in Ixs:
        I, x = Ix
        result += csr_array(
            (x,
                (D.flatten(),
                 [I_columns.tolist().index(i) for i in I.flatten()])
             ),
            shape=shape
        )

    if len(np.nonzero(g[I_columns])[0]):
        raise ValueError('g not correct, defines not removed DOFs')
    return result, g, np.setdiff1d(np.arange(D.N), D)


def combine(M1, M2):
    T1, g1, I1 = M1
    T2, g2, I2 = M2

    return (
        T1[:, np.isin(I1, I2)]
        @ T2[np.intersect1d(I1, I2), :][:, np.isin(I2, I1)],
        g1 + g2
        )


def apply(A, f, T, g):
    return T.T@A@T, T.T@(f-A@g)


def restore(y, T, g):
    return T@y+g


if __name__ == "__main__":
    from skfem import (MeshTri, ElementTriP1, Basis,
                       FacetBasis, asm, BilinearForm, solve, condense)
    from skfem.models import laplace, unit_load

    m = MeshTri().refined(6)
    e = ElementTriP1()

    basis = Basis(m, e)
    A = asm(laplace, basis)
    f = asm(unit_load, basis)

    # Penalty method

    fbases = [
        FacetBasis(m, e, facets='left'),
        FacetBasis(m, e, facets='right'),
    ]

    @BilinearForm
    def penalty(u, v, w):
        u1 = (w.idx[0] == 0) * u
        u2 = (w.idx[0] == 1) * u
        v1 = (w.idx[1] == 0) * v
        v2 = (w.idx[1] == 1) * v
        ju = u1 - 2 * u2
        jv = v1 - 2 * v2
        return 1. / 1e-2 * ju * jv

    B = asm(penalty, fbases, fbases)

    y1 = solve(
        *condense(A + B, f,
                  D=basis.get_dofs({'top', 'bottom'}),
                  x=basis.ones()
                  )
        )

    basis.plot(y1, colorbar=True, shading='gouraud', levels=5).show()

    # Master-Slave constraint

    T1, g1, I1 = multipoint(
        basis.get_dofs('left'),
        (basis.get_dofs('right'), [2]*len(basis.get_dofs('right').flatten()))
    )

    g2 = basis.zeros()
    g2[basis.get_dofs('top')] = 1
    T2, g2, I2 = multipoint(basis.get_dofs('top'), g=g2)

    g3 = basis.zeros()
    g3[basis.get_dofs('bottom')] = 1
    T3, g3, I3 = multipoint(basis.get_dofs('bottom'), g=g3)

    T, g = combine((T1, g1, I1), (T2, g2, I2))
    T, g = combine((T, g, np.intersect1d(I1, I2)), (T3, g3, I3))

    y2 = restore(solve(*apply(A, f, T, g)), T, g)

    basis.plot(y2, colorbar=True, shading='gouraud', levels=5).show()

    # Compare

    assert np.all(np.isclose(y1, y2, atol=2e-2))
