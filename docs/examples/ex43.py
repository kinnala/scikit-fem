r"""# Hyperelasticity

The strain energy density function per unit undeformed volume of the
isotropic hyperelastic Neo-Hookean material formulation is given as

..  math::

    \psi(\boldsymbol{F}) = \frac{\mu}{2} ( I_1 - 3) - \mu \ln(J) +
    \frac{\lambda}{2} \ln^2(J)


by the first invariant of the right Cauchy-Green deformation tensor
and the determinant of the deformation gradient (volume change).

..  math::

    I_1 &= \text{tr}(\boldsymbol{F}^T \boldsymbol{F}) =
    \boldsymbol{F} : \boldsymbol{F}

    J &= \det(\boldsymbol{F})


The variation of the strain energy function leads to

..  math::

    \delta\psi(\boldsymbol{F}) = \frac{\mu}{2} \delta I_1 - \mu \delta\ln(J)
    + \lambda \ln(J) \delta\ln(J)


with

..  math::

    \delta I_1 &= 2 \boldsymbol{F} : \delta\boldsymbol{F}

    \delta \ln(J) &= \boldsymbol{F}^{-T} : \delta\boldsymbol{F}


which finally results in

..  math::

    \delta\psi(\boldsymbol{F}) = \mu \boldsymbol{F} : \delta\boldsymbol{F} -
    (\lambda \ln(J) - \mu) \boldsymbol{F}^{-T} : \delta\boldsymbol{F}

The linearization of the variation leads to

..  math::

    \Delta\delta\psi(\boldsymbol{F}) = \mu \Delta\boldsymbol{F} :
    \delta\boldsymbol{F} + (\lambda \ln(J) - \mu)
    \Delta\boldsymbol{F}^{-T} : \delta\boldsymbol{F}
    + \lambda \left( \boldsymbol{F}^{-T} : \delta\boldsymbol{F} \right)
    \left( \boldsymbol{F}^{-T} : \Delta\boldsymbol{F} \right)

with

..  math::

    \Delta\boldsymbol{F}^{-T} = -\boldsymbol{F}^{-T} \Delta\boldsymbol{F}^{T}
    \boldsymbol{F}^{-T}

which may be alternatively formulated by expressions of the trace:

..  math::

    \Delta\delta\psi(\boldsymbol{F}) = \mu \Delta\boldsymbol{F} :
    \delta\boldsymbol{F} + (\lambda \ln(J) - \mu)
        \text{tr}(\Delta\boldsymbol{F} \boldsymbol{F}^{-1}
                  \delta\boldsymbol{F} \boldsymbol{F}^{-1})
    + \lambda \text{tr}(\delta\boldsymbol{F} \boldsymbol{F}^{-1})
        \text{tr}(\Delta\boldsymbol{F} \boldsymbol{F}^{-1})

"""
import numpy as np

from skfem import *
from skfem.helpers import grad, identity, ddot, det, transpose, inv, trace, mul

# note: rough mesh to make tests fast
mesh = MeshHex.init_tensor(
    np.linspace(0, 1, 10),
    np.linspace(-.1, .1, 3),
    np.linspace(-.1, .1, 3),
).with_boundaries({
    'left': lambda x: x[0] == 0.,
    'right': lambda x: x[0] == 1.,
})
element = ElementVector(ElementHex1())
basis = Basis(mesh, element, intorder=1)

mu, lmbda = 1., 2.

def deformation_gradient(w):
    dudX = grad(w["displacement"])
    F = dudX + identity(dudX)
    return F, inv(F)

@LinearForm
def L(v, w):
    F, iF = deformation_gradient(w)
    dF = grad(v)
    lnJ = np.log(det(F))
    return mu * ddot(F, dF) + (lmbda * lnJ - mu) * ddot(transpose(iF), dF)

@BilinearForm
def a(u, v, w):
    F, iF = deformation_gradient(w)
    DF = grad(u)
    dF = grad(v)
    dFiF = mul(dF, iF)
    DFiF = mul(DF, iF)
    tr_DFiF_dFiF = ddot(transpose(dFiF), DFiF)
    lnJ = np.log(det(F))
    return (mu * ddot(DF, dF) - (lmbda * lnJ - mu) * tr_DFiF_dFiF
            + lmbda * trace(dFiF) * trace(DFiF))

u  = basis.zeros()
du = basis.zeros()
uv = basis.interpolate(u)

dofs = basis.get_dofs('right')
dirichlet = basis.get_dofs({'right', 'left'})

f = L.assemble(basis, displacement=uv)
K = a.assemble(basis, displacement=uv)

right = -0.1
tol = 1e-10
nsteps = 7

for step in range(nsteps):
    for iteration in range(10):
        c = (step + 1.) / nsteps

        du_D = u.copy()
        x, y, z = basis.doflocs[:, dofs.nodal['u^1']]
        du_D[dofs.nodal['u^1']] = c * right - du_D[dofs.nodal['u^1']]
        du_D[dofs.nodal['u^2']] = (
            y * np.cos(c * np.pi) - z * np.sin(c * np.pi) - y
            - du_D[dofs.nodal['u^2']]
        )
        du_D[dofs.nodal['u^3']] = (
            y * np.sin(c * np.pi) + z * np.cos(c * np.pi) - z
            - du_D[dofs.nodal['u^3']]
        )

        du = solve(*condense(K, -f, x=du_D, D=dirichlet))
        norm_du = np.linalg.norm(du)
        u += du

        uv = basis.interpolate(u)

        f = L.assemble(basis, displacement=uv)
        K = a.assemble(basis, displacement=uv)

        print(1 + iteration, norm_du)

        if norm_du < tol:
            break


if __name__ == '__main__':
    (mesh.translated(u[basis.nodal_dofs])
         .draw('vedo', point_data={'uy': u[basis.nodal_dofs[1]]})
         .show())
