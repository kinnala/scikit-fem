from skfem import *
from skfem.models.poisson import unit_load
import numpy as np

m = MeshTri.init_symmetric()
m.refine(3)

e = ElementTriMorley()
ib = InteriorBasis(m, e)


@BilinearForm
def bilinf(u, v, w):
    from skfem.helpers import dd, ddot, trace, eye
    d = 0.1
    E = 200e9
    nu = 0.3

    def C(T):
        return E / (1 + nu) * (T + nu / (1 - nu) * eye(trace(T), 2))

    return d**3 / 12.0 * ddot(C(dd(u)), dd(v))


K = asm(bilinf, ib)
f = 1e6 * asm(unit_load, ib)

dofs = ib.find_dofs({
    'left':  m.facets_satisfying(lambda x: x[0] == 0),
    'right': m.facets_satisfying(lambda x: x[0] == 1),
    'top':   m.facets_satisfying(lambda x: x[1] == 1),
})

D = np.concatenate((
    dofs['left'].nodal['u'],
    dofs['left'].facet['u_n'],
    dofs['right'].nodal['u'],
    dofs['top'].nodal['u'],
))

x = solve(*condense(K, f, D=D))

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import *
    ax = draw(m)
    plot(ib, x, ax=ax, shading='gouraud', colorbar=True, Nrefs=2)
    savefig(splitext(argv[0])[0] + '_solution.png')
