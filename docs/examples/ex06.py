from skfem import *
from skfem.models.poisson import laplace, unit_load

m = MeshQuad()
m.refine(2)

e1 = ElementQuad1()
e = ElementQuad2()
mapping = MappingIsoparametric(m, e1)
ib = InteriorBasis(m, e, mapping, 4)

K = asm(laplace, ib)

f = asm(unit_load, ib)

x = solve(*condense(K, f, D=ib.get_dofs()))

M, X = ib.refinterp(x, 3)

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import *
    ax = draw(m)
    plot(M, X, ax=ax, shading='gouraud', edgecolors='')
    savefig(splitext(argv[0])[0] + '_solution.png')
