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


print('| Degrees-of-freedom | Time spent in assembly (s) | Time spent in linear solve (s) |')
print('| --- | --- | --- |')


for N in range(8, 30, 2):
    
    m = pre(N)

    def assembler(m):
        basis = InteriorBasis(m, ElementTetP1())
        return (asm(laplace, basis),
                asm(unit_load, basis),
                m.boundary_nodes())

    A, b, D = assembler(m)

    assemble_time = timeit(lambda: assembler(m), number=3) / 3.

    def solver(A, b):
        return solve(*condense(A, b, D=D))

    solve_time = timeit(lambda: solver(A, b), number=3) / 3.

    print('| {} | {} | {} |'.format(len(b), assemble_time, solve_time))
