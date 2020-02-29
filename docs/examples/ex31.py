from skfem import *
from skfem.models.poisson import laplace, mass
import numpy as np

# create mesh and element for the finite element approximation
m = MeshQuad()
m.refine()
e = ElementQuadP(5)

# create quadratic mesh for the local-to-global mapping
E = ElementQuad2()
mapping_basis = InteriorBasis(m, E)
M = MeshQuad.from_basis(mapping_basis)

# deform the quadratic mesh
f1 = M.facets_satisfying(lambda x: (np.abs(x[0] - .5) == .5) * (x[1] > .5))
f2 = M.facets_satisfying(lambda x: (np.abs(x[0] - .5) == .5) * (x[1] < .5))
f3 = M.facets_satisfying(lambda x: (np.abs(x[1] - .5) == .5) * (x[0] > .5))
f4 = M.facets_satisfying(lambda x: (np.abs(x[1] - .5) == .5) * (x[0] < .5))
M.p[0, mapping_basis.facet_dofs[:, f1]] += 0.1
M.p[0, mapping_basis.facet_dofs[:, f2]] -= 0.1
M.p[1, mapping_basis.facet_dofs[:, f3]] -= 0.1
M.p[1, mapping_basis.facet_dofs[:, f4]] += 0.1

# create mapping for the finite element approximation and assemble
mapping = MappingIsoparametric(M, E)
basis = InteriorBasis(m, e, mapping)

A = asm(laplace, basis)
M = asm(mass, basis)

L, x = solve(*condense(A, M, D=basis.find_dofs()), k=8)


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv
    name = splitext(argv[0])[0]

    from skfem.visuals.matplotlib import *
    ax = draw(m)
    plot(basis, x[:, 6], Nrefs=6, ax=ax)
    savefig(f'{name}_eigenmode.png')
