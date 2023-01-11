import numpy as np
from scipy.sparse import csr_array

if __name__ == '__main__':
    from skfem import *
    from skfem.models import laplace, unit_load

    m = MeshTri().refined(6)
    e = ElementTriP1()

    basis = Basis(m, e)

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

    A = asm(laplace, basis)
    B = asm(penalty, fbases, fbases)
    f = asm(unit_load, basis)

    y = solve(*condense(A + B, f, D=basis.get_dofs({'top', 'bottom'})))

    basis.plot(y, colorbar=True, shading='gouraud', levels=5, ax=m.draw()).show()


def multipoint(Dx, *Ixs):
    D,x = Dx
    if x is None:
        x = np.ones((D.N-len(D.flatten()),))
    shape = D.N, D.N-len(D.flatten())

    D_rows = np.arange(D.N-len(D.flatten()))
    D_columns = np.setdiff1d(np.arange(D.N), D)

    result = csr_array(
        (x, (D_columns, D_rows)), shape=shape
    )

    for Ix in Ixs:
        I, x = Ix
        result+= csr_array(
            (x, (D.flatten(),[D_columns.tolist().index(i) for i in I.flatten()])), shape=shape
        )

    return result, np.setdiff1d(np.arange(D.N), D)

def combine(M1, M2):
    T1, I1 = M1
    T2, I2 = M2

    return T1[:,np.isin(I1, I2)]@T2[np.intersect1d(I1,I2),:][:,np.isin(I2, I1)]

if __name__ == "__main__":
    T1, I1=multipoint((basis.get_dofs('left'), None), (basis.get_dofs('right'), [2]*len(basis.get_dofs('right').flatten())))
    T2, I2=multipoint((basis.get_dofs('top') + basis.get_dofs('bottom'),None))

    T = combine((T1,I1),(T2,I2))

    A = asm(laplace, basis)
    f = asm(unit_load, basis)

    y = solve(T.T@A@T, T.T@f)

    basis.plot(T@y, colorbar=True, shading='gouraud', levels=5, ax=m.draw()).show()

