from skfem import *

m = MeshLine()
m.refine(3)
e = ElementLineHermite()
basis = InteriorBasis(m, e)

@BilinearForm
def bilinf(u, v, w):
    from skfem.helpers import dd, ddot
    return ddot(dd(u), dd(v))

@LinearForm
def linf(v, w):
    return 1.0 * v

A = asm(bilinf, basis)
f = asm(linf, basis)

D = basis.find_dofs({
    'left': m.facets_satisfying(lambda x: x[0] == 0),
})

x = solve(*condense(A, f, D=D))

# compare to analytical solution
err = max(x[basis.nodal_dofs[0]]) - 1. / 8.
print(err)

if __name__ == '__main__':

    from os.path import splitext
    from sys import argv
    name = splitext(argv[0])[0]

    from skfem.visuals.matplotlib import *
    plot(basis, x, Nrefs=3)
    savefig(f'{name}_solution.png')
