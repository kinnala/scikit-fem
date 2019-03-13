from pathlib import Path
from skfem import *
from skfem.models.poisson import laplace, unit_load
import numpy as np

m = MeshQuad()
m.refine(2)

e1 = ElementQuad1()
e = ElementQuad2()
mapping = MappingIsoparametric(m, e1)
ib = InteriorBasis(m, e, mapping, 4)

K = asm(laplace, ib)

f = asm(unit_load, ib)

D = ib.get_dofs().all()
x = np.zeros(ib.N)
I = ib.complement_dofs(D)

x[I] = solve(*condense(K, f, D=D))

M, X = ib.refinterp(x, 3)

if __name__ == "__main__":
    ax = m.draw()
    M.plot(X, smooth=True, edgecolors='', ax=ax)
    M.savefig(Path(__file__).stem + '_solution.png')
