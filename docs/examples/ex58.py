r"""Lid-driven cavity at high Re.

Steady incompressible Navier-Stokes equations in the unit square,

.. math::
    -\tfrac{1}{Re} \Delta u + (u \cdot \nabla) u + \nabla p = 0, \quad
    \nabla \cdot u = 0,

with :math:`u = (1, 0)` on the lid :math:`y = 1` and :math:`u = 0` on
the other walls.  Discretized with the quadrilateral Q2-Q1 pair and
solved with Newton's method: given :math:`u_0`, find :math:`(u, p)`
s.t.

.. math::
    \tfrac{1}{Re} (\nabla u, \nabla v) + ((u_0 \cdot \nabla) u
    + (u \cdot \nabla) u_0, v) - (p, \nabla \cdot v)
    = ((u_0 \cdot \nabla) u_0, v), \quad (\nabla \cdot u, q) = 0.

"""
import numpy as np
from skfem import *
from skfem.helpers import ddot, div, dot, grad, mul
from skfem.models import laplace

# from skfem.experimental.cforms import CBilinearForm, CLinearForm
# BilinearForm = CBilinearForm(cache_sparsity=True)
# LinearForm = CLinearForm

# mesh graded towards the walls
s = (1. - np.cos(np.pi * np.linspace(0., 1., 30))) / 2.
mesh = MeshQuad.init_tensor(s, s).with_defaults()
ubasis = Basis(mesh, ElementVector(ElementQuad2()))
pbasis = ubasis.with_element(ElementQuad1())
Nu = ubasis.N


@BilinearForm
def jacobian(u, v, w):
    return (ddot(grad(u), grad(v)) / w.Re
            + dot(mul(grad(u), w.u0) + mul(grad(w.u0), u), v))


@BilinearForm
def divergence(u, q, w):
    return div(u) * q


@LinearForm
def convection(v, w):
    return dot(mul(grad(w.u0), w.u0), v)


B = divergence.assemble(ubasis, pbasis)

# velocity fixed on all walls, pressure fixed to zero at a point
D = np.concatenate((ubasis.get_dofs().all(), [ubasis.N]))
x = np.zeros(ubasis.N + pbasis.N)  # (u, p)

# u_x = 1 on the lid
lid = ubasis.get_dofs('top').all('u^1')
x[lid] = 1.


for Re in [100, 400]:  # add more Reynolds numbers here
    for itr in range(20):
        u0 = x[:ubasis.N]
        K = bmat([[jacobian.assemble(ubasis, u0=u0, Re=Re), -B.T],
                  [-B, None]]).tocsr()
        f = np.concatenate((convection.assemble(ubasis, u0=u0),
                            np.zeros(pbasis.N)))
        xnew = solve(*condense(K, f, x=x, D=D))
        step = np.linalg.norm(xnew - x) / np.linalg.norm(xnew)
        x = xnew
        print(step)
        if step < 1e-10:
            break
    else:
        raise RuntimeError("Newton did not converge at Re={}".format(Re))
    print("Re={:5d}: {} Newton iterations".format(Re, itr + 1))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skfem.visuals.matplotlib import plot
    # calculate the streamfunction:
    # -Laplace(psi) = curl(u), psi = 0 on the walls
    psibasis = ubasis.with_element(ElementQuad2())

    @LinearForm
    def vorticity(v, w):
        return (grad(w.u)[1, 0] - grad(w.u)[0, 1]) * v

    A = laplace.assemble(psibasis)
    b = vorticity.assemble(psibasis, u=ubasis.interpolate(x[:ubasis.N]))
    psi = solve(*condense(A, b, D=psibasis.get_dofs()))

    # plot the streamfunction
    ax = plot(psibasis, psi, Nrefs=2, shading='gouraud', colorbar=True,
              levels=[-0.1, -0.05, -0.01, -0.001, -0.0001, -0.00001, 0.00001, 0.0001])
    ax.set_title("Streamfunction, Re = {}".format(Re))
    plt.savefig('ex58.png')
    plt.show()
