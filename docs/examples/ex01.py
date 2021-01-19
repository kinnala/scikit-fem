from skfem import *
from skfem.models.poisson import laplace, unit_load

m = MeshTri().refined(4)

e = ElementTriP1()
basis = InteriorBasis(m, e)

A = asm(laplace, basis)
b = asm(unit_load, basis)

x = solve(*condense(A, b, I=m.interior_nodes()))

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import plot, savefig
   
    plot(m, x, shading='gouraud', colorbar=True)
    savefig(splitext(argv[0])[0] + '_solution.png')
