"""Contact problem."""
from skfem import *
from skfem.autodiff import *
from skfem.autodiff.helpers import *
from skfem.supermeshing import intersect, elementwise_quadrature
import jax.numpy as jnp
import numpy as np


m1 = (MeshQuad
      .init_tensor(np.linspace(0, 5, 60), np.linspace(0, 0.25, 5))
      .with_defaults())
m2 = (MeshQuad
      .init_tensor(np.linspace(0, 5, 51), np.linspace(-0.5, -0.25, 4))
      .with_defaults())

e1 = ElementVector(ElementQuad1())
e2 = ElementVector(ElementQuad1())

basis1 = Basis(m1, e1)
basis2 = Basis(m2, e2)

# forms are based on the energy minimization problem

@NonlinearForm(hessian=True)
def g(u1, u2, w):
    eps = 1e-2
    return 1 / eps * jnp.minimum(0.25 + u1[1] - u2[1], 0) ** 2


@NonlinearForm(hessian=True)
def J1(u, w):
    eps = 1/2 * (grad(u) + transpose(grad(u)) + mul(transpose(grad(u)), grad(u)))
    sig = 2 * eps + eye(trace(eps), 2)
    return 1/2 * ddot(sig, eps)


@NonlinearForm(hessian=True)
def J2(u, w):
    eps = 1/2 * (grad(u) + transpose(grad(u)) + mul(transpose(grad(u)), grad(u)))
    sig = 2 * eps + eye(trace(eps), 2)
    # apply body force for domain 2
    return 1/2 * ddot(sig, eps) - 1e-2 * u[1] - 5e-2 * u[0]


x = np.zeros(basis1.N + basis2.N)
m1defo = m1.copy()
m2defo = m2.copy()

for itr in range(40):

    # use deformed mesh for mapping between domain 1 and 2
    m1t, orig1 = m1defo.trace('bottom',
                              mtype=MeshLine,
                              project=lambda p: np.array(p[0]))
    m2t, orig2 = m2defo.trace('top',
                              mtype=MeshLine,
                              project=lambda p: np.array(p[0]))
    m12, t1, t2 = intersect(m1t, m2t)

    fbasis1 = FacetBasis(m1, e1,
                         quadrature=elementwise_quadrature(m1t, m12, t1),
                         facets=orig1[t1])
    fbasis2 = FacetBasis(m2, e2,
                         quadrature=elementwise_quadrature(m2t, m12, t2),
                         facets=orig2[t2])
    fbasis = fbasis1 * fbasis2

    # assemble jacobians for 1 and 2 separately and create a block matrix
    jac1, rhs1 = J1.assemble(basis1, x=x[:basis1.N])
    jac2, rhs2 = J2.assemble(basis2, x=x[basis1.N:])
    jac = bmat([[jac1, None],
                [None, jac2]])
    rhs = np.concatenate((rhs1, rhs2))

    # g assembled using fbasis1 * fbasis2 has automatically the correct shape
    jacg, rhsg = g.assemble(fbasis, x=x)
    jac = jac + jacg
    rhs = rhs + rhsg
    
    dx = solve(*enforce(jac, np.array(rhs), D=np.concatenate((
        basis1.get_dofs({'left', 'right'}).all(),
        basis2.get_dofs({'left', 'right'}).all() + basis1.N,
    ))))
    x += 0.95 * dx  # regularization
    print(np.linalg.norm(dx))
    if np.linalg.norm(dx) < 1e-8:
        break
    (x1, _), (x2, _) = fbasis.split(x)
    m1defo = m1.translated(x1[basis1.nodal_dofs])
    m2defo = m2.translated(x2[basis2.nodal_dofs])

# postprocessing: calculate stress and visualize

# calculate stress
u1 = basis1.interpolate(x1)
eps1 = 1/2 * (grad(u1) + transpose(grad(u1)) + mul(transpose(grad(u1)), grad(u1)))
sig1 = 2 * eps1 + eye(trace(eps1), 2)

u2 = basis2.interpolate(x2)
eps2 = 1/2 * (grad(u2) + transpose(grad(u2)) + mul(transpose(grad(u2)), grad(u2)))
sig2 = 2 * eps2 + eye(trace(eps2), 2)

pbasis1 = basis1.with_element(ElementQuad0())
pbasis2 = basis2.with_element(ElementQuad0())

# basis with deformed mesh for convenient plotting
pbasis1defo = Basis(m1defo, ElementQuad0())
pbasis2defo = Basis(m2defo, ElementQuad0())

sigyy1 = pbasis1.project(np.array(sig1[1, 1]))
sigyy2 = pbasis2.project(np.array(sig2[1, 1]))

if __name__ == "__main__":

    ax = m1defo.draw()
    m2defo.draw(ax=ax)
    ax.set_xlim(0.5, 4.5)
    pbasis1defo.plot(pbasis1.project(np.array(sig1[1, 1])),
                     vmax=0.0,
                     vmin=-0.01,
                     ax=ax,
                     cmap='viridis')
    pbasis2defo.plot(pbasis2.project(np.array(sig2[1, 1])),
                     vmax=0.0,
                     vmin=-0.01,
                     ax=ax,
                     colorbar={'orientation': 'horizontal',
                               'label': r'$\sigma_{yy}$'},
                     cmap='viridis').show()
