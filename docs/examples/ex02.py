from skfem import *
import numpy as np

m = MeshTri.init_symmetric()
m.refine(3)

e = ElementTriMorley()
ib = InteriorBasis(m, e)

@bilinear_form
def bilinf(u, du, ddu, v, dv, ddv, w):
    d = 0.1
    E = 200e9
    nu = 0.3

    def C(T):
        trT = T[0, 0] + T[1, 1]
        return E / (1. + nu) * \
            np.array([[T[0, 0] + nu / (1. - nu) * trT, T[0, 1]],
                      [T[1, 0], T[1, 1] + nu / (1. - nu) * trT]])

    def Eps(ddw):
        return np.array([[ddw[0][0], ddw[0][1]],
                         [ddw[1][0], ddw[1][1]]])

    def ddot(T1, T2):
        return (T1[0, 0] * T2[0, 0] +
                T1[0, 1] * T2[0, 1] +
                T1[1, 0] * T2[1, 0] +
                T1[1, 1] * T2[1, 1])

    return d**3 / 12.0 * ddot(C(Eps(ddu)), Eps(ddv))

@linear_form
def linf(v, dv, ddv, w):
    return 1e6 * v

K = asm(bilinf, ib)
f = asm(linf, ib)

dofs = ib.get_dofs({
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
