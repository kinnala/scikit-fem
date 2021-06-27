"""A simple performance test.

This script is used to generate the table in README.md.

"""
from timeit import timeit
import numpy as np
from skfem import *
from skfem.models.poisson import laplace, unit_load


def pre(N=3):
    m = MeshTet.init_tensor(*(3 * (np.linspace(0., 1., N),)))
    return m


print('| Degrees-of-freedom | Assembly (s) | Linear solve (s) |')
print('| --- | --- | --- |')


def assembler(m):
    basis = Basis(m, ElementTetP1())
    return (
        asm(laplace, basis),
        asm(unit_load, basis),
    )


for k in range(6, 21):

    m = pre(int(2 ** (k / 3)))

    assemble_time = timeit(lambda: assembler(m), number=1)

    A, b = assembler(m)
    D = m.boundary_nodes()

    if A.shape[0] > 1e5:
        solve_time = np.nan
    else:
        solve_time = timeit(lambda: solve(*condense(A, b, D=D)), number=1)

    print('| {} | {:.5f} | {:.5f} |'.format(len(b), assemble_time, solve_time))

    del A
    del b
    del D
    del m
