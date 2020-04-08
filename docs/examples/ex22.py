from skfem import *
from skfem.models.poisson import laplace
import numpy as np

m = MeshTri.init_lshaped()
m.refine(2)
e = ElementTriP1()

def load_func(x, y):
    return 1.0

@LinearForm
def load(v, w):
    x, y = w.x
    return load_func(x, y) * v

def eval_estimator(m, u):    
    # interior residual
    basis = InteriorBasis(m, e)
    
    @functional
    def interior_residual(w):
        h = w.h
        x, y = w.x
        return h**2 * load_func(x, y)**2

    eta_K = interior_residual.elemental(basis, w=basis.interpolate(u))
    
    # facet jump
    fbasis = [FacetBasis(m, e, side=i) for i in [0, 1]]   
    w = [fbasis[i].interpolate(u) for i in [0, 1]]
    
    @functional
    def edge_jump(w):
        h = w.h
        n = w.n
        du1, du2 = w.dw
        return h * ((du1[0] - du2[0])*n[0] +\
                    (du1[1] - du2[1])*n[1])**2

    eta_E = edge_jump.elemental(fbasis[0], w=w)
    
    tmp = np.zeros(m.facets.shape[1])
    np.add.at(tmp, fbasis[0].find, eta_E)
    eta_E = np.sum(0.5*tmp[m.t2f], axis=0)
    
    return eta_K + eta_E

if __name__ == "__main__":
    from skfem.visuals.matplotlib import draw, plot, show
    draw(m)

for itr in range(10): # 10 adaptive refinements
    if itr > 1:
        m.refine(adaptive_theta(eval_estimator(m, u)))
        
    basis = InteriorBasis(m, e)
    
    K = asm(laplace, basis)
    f = asm(load, basis)
    u = np.zeros_like(f)
    
    I = m.interior_nodes()
    u = solve(*condense(K, f, I=I))

if __name__ == "__main__":
    draw(m)
    plot(m, u, shading='gouraud')
    show()
