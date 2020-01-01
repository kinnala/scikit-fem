from skfem import *
import numpy as np

m = MeshTri()
m.refine(5)

@bilinear_form
def jacobian(u, du, v, dv, w):
    w, dw = w.w, w.dw
    return 1.0/np.sqrt(1.0 + dw[0]**2 + dw[1]**2)*(du[0]*dv[0] + du[1]*dv[1])\
           -(2.0*du[1]*dw[1] + 2.0*du[0]*dw[0])*(dw[1]*dv[1] + dw[0]*dv[0])\
           /(2.0*(1 + dw[1]**2 + dw[0]**2)**(3./2.))

@linear_form
def rhs(v, dv, w):
    w, dw = w.w, w.dw
    return 1.0/np.sqrt(1.0 + dw[0]**2 + dw[1]**2)*(dw[0]*dv[0] + dw[1]*dv[1])

basis = InteriorBasis(m, ElementTriP1())

x = np.zeros(basis.N)

I = m.interior_nodes()
D = m.boundary_nodes()
x[D] = np.sin(np.pi * m.p[0, D]) 

for itr in range(100):
    w = basis.interpolate(x)
    J = asm(jacobian, basis, w=w)
    F = asm(rhs, basis, w=w)
    x_prev = x.copy()
    x += 0.7 * solve(*condense(J, -F, I=I))
    if np.linalg.norm(x - x_prev) < 1e-8:
        break
    if __name__ == "__main__":
        print(np.linalg.norm(x - x_prev))

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot3
    plot3(m, x)
    show()
