"""Hertz contact with Nitsche's method

A cylinder (m1) is pressed against an elastic block (m2):

- linear elasticity, small deformations
- contact pairing in the reference configuration
- quadratic mesh for accurate contact surface
- one-sided Nitsche method

"""
from skfem import *
from skfem.autodiff import *
from skfem.autodiff.helpers import *
from skfem.supermeshing import mortar_quadrature
from skfem.models.elasticity import lame_parameters
import numpy as np


R = 1.0  # cylinder radius
E, nu = 1000.0, 0.3
lam, mu = lame_parameters(E, nu)
alpha = 1e-2  # Nitsche parameter
delta = 3e-3  # prescribed displacement of the top
nrefs = 0

# mesh generation

def graded(L, n, beta=3.5):
    """n + 1 graded nodes on [0, L]"""
    s = np.linspace(0, 1, n + 1)
    return L * np.sinh(beta * s) / np.sinh(beta)


# cylinder: bottom follows the circle y = R - sqrt(R^2 - x^2), top at y = 0.5
trans = lambda x: np.array([
    0 * x[0],
    (R - np.sqrt(R ** 2 - x[0] ** 2)) * (1 - x[1] / 0.5)
])
xs = np.concatenate((
    -graded(0.5, 10)[:0:-1],  # symmetric wrt x = 0
    graded(0.5, 10)
))
ys = graded(0.5, 8)
m1 = (MeshQuad2
      .from_mesh(MeshQuad.init_tensor(xs, ys).refined(nrefs))
      .with_defaults()
      .translated(trans))

# block [-1, 1] x [-1, 0]
xs = np.concatenate((
    -graded(1.0, 13, 4.0)[:0:-1],
    graded(1.0, 13, 4.0)
))
ys = -graded(1.0, 25, 4.0)[::-1]
m2 = (MeshQuad.init_tensor(xs, ys)
      .refined(nrefs)
      .with_defaults())

# define elements, bases and forms

e = ElementVector(ElementQuad2())
basis1, basis2 = Basis(m1, e), Basis(m2, e)
N1 = basis1.N


def sigma(u):
    eps = sym_grad(u)
    return 2 * mu * eps + lam * eye(trace(eps), 2)


@NonlinearForm(hessian=True)
def elastic_energy(u, w):
    return 1 / 2 * ddot(sigma(u), sym_grad(u))


def contact_pressure(u1, w):
    return -dot(mul(sigma(u1), w.n), w.n)


def gap(u1, u2, w):
    return dot(w.x2 + u2 - w.x - u1, w.n)


@NonlinearForm(hessian=True)
def nitsche(u1, u2, w):
    gamma = alpha * w.h / mu
    lam_u = contact_pressure(u1, w)
    return (gamma / 2 * np.maximum(lam_u - gap(u1, u2, w) / gamma, 0) ** 2
            - gamma / 2 * lam_u ** 2)


# contact pairing
q1, f1, q2, f2 = mortar_quadrature(m1, 'bottom',
                                   m2, 'top', max_dist=0.2, nsub=1)
fb1 = FacetBasis(m1, e, quadrature=q1, facets=f1)
fb2 = FacetBasis(m2, e, quadrature=q2, facets=f2)
fbasis = fb1 * fb2
X2 = fb2.global_coordinates()  # master points paired with slave points

# boundary conditions
D = np.concatenate((basis1.get_dofs('top').all(),
                    basis2.get_dofs('bottom').all() + N1))
x = fbasis.zeros()
x[basis1.get_dofs('top').all('u^2')] = -delta

# Hessian of energy is assembled once
K11 = elastic_energy.assemble(basis1, x=np.zeros(N1))[0]
K22 = elastic_energy.assemble(basis2, x=np.zeros(basis2.N))[0]

K = bmat([[K11, None],
          [None, K22]])

# semismooth Newton
for itr in range(30):
    # assemble Jacobian of the contact term
    jac, rhs = nitsche.assemble(fbasis, x=x, x2=X2)
    dx = solve(*enforce(K + jac, rhs - K @ x, D=D))
    x += dx
    print(f"iteration {itr + 1}: |dx| = {np.linalg.norm(dx):.2e}")
    if np.linalg.norm(dx) < 1e-12:
        break

# postprocessing

def von_mises(basis, u):
    s = sigma(basis.interpolate(u))
    s33 = nu * (s[0, 0] + s[1, 1])
    vm = np.sqrt(0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s33) ** 2
                        + (s33 - s[0, 0]) ** 2) + 3 * s[0, 1] ** 2)
    return basis.with_element(ElementQuad1()).project(vm)

# split solution and calculate von Mises stress
x1, x2 = x[:N1], x[N1:]
mises1, mises2 = von_mises(basis1, x1), von_mises(basis2, x2)
mises = np.concatenate((mises1, mises2))

# show both bodies in the same visualization
basis_mises = Basis(MeshQuad.from_mesh(m1) + m2, ElementQuad1())

# helper to plot contact force
@Functional
def multiplier_plotter(w):
    gamma = alpha * w.h / mu
    lamh = np.maximum(contact_pressure(w.u[0], w)
                      - gap(w.u[0], w.u[1], w) / gamma, 0)

    # mesh for plotting multiplier at every quadrature point
    global mesh_mult, mult
    mesh_mult = MeshLine(w.x[0].flatten())
    mult = lamh.flatten()

    # return per quadrature point array for integration
    return lamh * w.n[1]


print("Total contact force: {}".format(
    multiplier_plotter.assemble(fbasis, u=x, x2=X2)
))


def visualize():
    return basis_mises.plot(mises, 'web', nrefs=3, cmap='turbo')

if __name__ == "__main__":
    visualize().show()
