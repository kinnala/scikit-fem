from sys import argv

import numpy as np

import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc

import ex16


def petsc_mat(A):
    return PETSc.Mat().createAIJ(
        size=A.shape,
        csr=(A.indptr, A.indices, A.data))


slepc4py.init(argv)

E = SLEPc.EPS().create()

E.setOperators(*map(petsc_mat, [ex16.L, ex16.M]))

E.setType(E.Type.ARNOLDI)
E.setDimensions(2 * len(ex16.ks), PETSc.DECIDE)
E.setWhichEigenpairs(E.Which.SMALLEST_MAGNITUDE)
E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
E.setFromOptions()

E.solve()

if __name__ == "__main__":

    from os.path import splitext
    
    for n, k in enumerate(ex16.ks):
        print('{:2d}  {:9.6f}  {:9.6f}'.format(
            n * (n + 1), k, np.real_if_close(E.getEigenvalue(n))))
