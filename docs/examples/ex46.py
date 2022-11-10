"""Waveguide cutoff analysis."""

import numpy as np
import matplotlib.pyplot as plt
from skfem import *
from skfem.helpers import *


# two different mesh and element types
mtype_elem = [
    (MeshTri, ElementTriN0() * ElementTriP1()),
    (MeshQuad, ElementQuadN0() * ElementQuad1()),
]


for mtype, elem in mtype_elem:
    xs = np.linspace(0, 1, 49) ** 0.9
    ys = np.linspace(0, .5, 20)
    mesh = mtype().init_tensor(xs, ys)
    basis = Basis(mesh, elem)

    epsilon = lambda x: 1. + 0. * x[0]
    # epsilon = lambda x: 3 * (x[1] < 0.25) + 1
    one_over_u_r = 1


    @BilinearForm
    def aform(E, lam, v, mu, w):
        return one_over_u_r * curl(E) * curl(v)


    @BilinearForm
    def gauge(E, lam, v, mu, w):
        # set div E = 0 using a Lagrange multiplier
        return dot(grad(lam), v) + dot(E, grad(mu))


    @BilinearForm
    def bform(E, lam, v, mu, w):
        return epsilon(w.x) * dot(E, v)


    A = aform.assemble(basis)
    B = bform.assemble(basis)
    C = gauge.assemble(basis)

    lams, xs = solve(*condense(A + C, B, D=basis.get_dofs()),
                     solver=solver_eigen_scipy_sym(k=6))


    # compare against analytical eigenvalues
    err1 = np.abs(lams[0] - np.pi ** 2)
    err2 = np.abs(lams[1] - 4. * np.pi ** 2)
    err3 = np.abs(lams[2] - 4. * np.pi ** 2)


    if __name__ == "__main__":
        print('TE10 error: {}'.format(err1))
        print('TE01 error: {}'.format(err2))
        print('TE20 error: {}'.format(err3))

        fig, axs = plt.subplots(4, 1)
        for itr in range(4):
            (E, Ebasis), (_, lambasis) = basis.split(xs[:, itr])
            Ei = Ebasis.interpolate(E)
            Emag = lambasis.project(np.sqrt(dot(Ei, Ei)))
            lambasis.plot(Emag, colorbar=True, ax=axs[itr])
        plt.show()
