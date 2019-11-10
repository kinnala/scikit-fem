from skfem import *

m = MeshTri()
m.refine(4)

e = ElementTriP1()
basis = InteriorBasis(m, e)

@bilinear_form
def laplace(u, du, v, dv, w):
    return du[0]*dv[0] + du[1]*dv[1]

@linear_form
def load(v, dv, w):
    return 1.0*v

A = asm(laplace, basis)
b = asm(load, basis)

x = solve(*condense(A, b, I=m.interior_nodes()))

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    
    m.plot(x, smooth=True, colorbar=True)
    m.savefig(splitext(argv[0])[0] + '_solution.png')
