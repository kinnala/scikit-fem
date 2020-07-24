from timeit import timeit

from skfem import *


@BilinearForm
def bilinf(u, v, w):
    return sum(u.grad * v.grad)


@LinearForm
def linf(v, w):
    return 1. * v


def pre(refs=0):
    m = MeshTet()
    m.refine(refs)
    return m


print('| Degrees-of-freedom | Assembly time (s) | Linear solve time (s) |')
print('| --- | --- | --- |')


for refs in [1, 2, 3, 4, 5]:
    
    m = pre(refs)

    def assembler(m):
        basis = InteriorBasis(m, ElementTetP1())
        return (asm(bilinf, basis),
                asm(linf, basis),
                m.boundary_nodes())

    A, b, D = assembler(m)

    assemble_time = timeit(lambda: assembler(m), number=3) / 3.

    def solver(A, b):
        return solve(*condense(A, b, D=D))

    solve_time = timeit(lambda: solver(A, b), number=3) / 3.

    print('| {} | {} | {} |'.format(len(b), assemble_time, solve_time))
