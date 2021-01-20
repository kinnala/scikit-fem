r"""Adaptive Poisson equation.

This example solves `ex01.py` adaptively in an L-shaped domain.
Using linear elements, the error indicators read

.. math::
   \eta_K^2 = h_K^2 \|f\|_{0,K}^2
for each element :math:`K`, and

.. math::
   \eta_E^2 = h_E \| [[\nabla u_h \cdot n ]] \|_{0,E}^2
for each edge :math:`E`.

"""
from skfem import *
from skfem.models.poisson import laplace
from skfem.helpers import grad
import numpy as np


m = MeshTri.init_lshaped().refined(2)
e = ElementTriP1()


def load_func(x, y):
    return 1.


@LinearForm
def load(v, w):
    x, y = w.x
    return load_func(x, y) * v


def eval_estimator(m, u):    
    # interior residual
    basis = InteriorBasis(m, e)
    
    @Functional
    def interior_residual(w):
        h = w.h
        x, y = w.x
        return h ** 2 * load_func(x, y) ** 2

    eta_K = interior_residual.elemental(basis, w=basis.interpolate(u))
    
    # facet jump
    fbasis = [InteriorFacetBasis(m, e, side=i) for i in [0, 1]]
    w = {'u' + str(i + 1): fbasis[i].interpolate(u) for i in [0, 1]}
    
    @Functional
    def edge_jump(w):
        h = w.h
        n = w.n
        dw1 = grad(w['u1'])
        dw2 = grad(w['u2'])
        return h * ((dw1[0] - dw2[0]) * n[0] +\
                    (dw1[1] - dw2[1]) * n[1]) ** 2

    eta_E = edge_jump.elemental(fbasis[0], **w)
    
    tmp = np.zeros(m.facets.shape[1])
    np.add.at(tmp, fbasis[0].find, eta_E)
    eta_E = np.sum(0.5*tmp[m.t2f], axis=0)
    
    return eta_K + eta_E

if __name__ == "__main__":
    from skfem.visuals.matplotlib import draw, plot, show
    draw(m)

for itr in range(9): # 9 adaptive refinements
    if itr > 0:
        m = m.refined(adaptive_theta(eval_estimator(m, u)))
        
    basis = InteriorBasis(m, e)
    
    K = asm(laplace, basis)
    f = asm(load, basis)
    
    I = m.interior_nodes()
    u = solve(*condense(K, f, I=I))

if __name__ == "__main__":
    draw(m)
    plot(m, u, shading='gouraud')
    show()
