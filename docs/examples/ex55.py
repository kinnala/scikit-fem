r"""Hyperelasticity in Updated-Lagrange formulation.

This example implements the hyperelastic problem of excercise 43 in the Updated Lagrange formulation.
The strain energy density function per unit undeformed volume of the
isotropic hyperelastic Neo-Hookean material is given by

..  math::

    \psi(\boldsymbol{F}) = \frac{\mu}{2} ( I_1 - 3) - \mu \ln(J) +
    \frac{\lambda}{2} \ln^2(J)

This time we formulate the first invariant in terms of the left Cauchy-Green deformation tensor
and the determinant of the deformation gradient (volume change).

..  math::

    I_1 &= \text{Tr}(\boldsymbol{F} \boldsymbol{F}^T) = \text{Tr}(\boldsymbol{b}) \\
    J   &= \det(\boldsymbol{F}) = \sqrt{\det(\boldsymbol{b})}

For the Updated Lagrange formulation we formulate the weak form in terms of the current configuration

..  math::

    0 = \int_{\omega} \frac{1}{J}\boldsymbol{\tau}(\vec{u}) : \vec{\nabla}(\vec{v}) \, dv

With the Kirchhoff stress 

..  math::

    \boldsymbol{\tau} = 2 \frac{\partial \psi}{\partial \boldsymbol{b}}\cdot\boldsymbol{b} 
                      = \mu \left[\boldsymbol{b}-\boldsymbol{1}\right]+\lambda\ln(J)\boldsymbol{1}

The linearization of the weak form in the current configuration can be derived as

..  math::

    \int_{\omega}
    \frac{1}{J}
    \vec{\nabla}(\vec{v})
    :\left[
    \frac{\partial \boldsymbol{\tau}}{\partial \boldsymbol{F}} \cdot \boldsymbol{F}^T
    -
    \left[\boldsymbol{\tau}\square\boldsymbol{1}\right]^{T_{R}}
    \right]:
    \vec{\nabla}(\Delta\vec{u})
    \, dv
    =
    \int_{\omega}
    \frac{1}{J}
    v_{i,j} \left[
     \frac{\partial \tau_{ij}}{\partial F_{kN}} F_{lN} - \tau_{il} \delta_{kj}
    \right] \Delta u_{k,l}
    \, dv

with the stiffness contribution

..  math::

    \left[\frac{\partial \boldsymbol{\tau}}{\partial \boldsymbol{F}}\right]_{ijkN}
    &=
    \mu\left[\delta_{ik} F_{jN}+\delta_{jk}F_{iN}\right] + \lambda \delta_{ij}F_{Nk}^{-1} \\
    \left[\frac{\partial \boldsymbol{\tau}}{\partial \boldsymbol{F}} \cdot \boldsymbol{F}^T\right]_{ijkl}
    &=
    \mu\left[\delta_{ik} b_{lj}+b_{il}\delta_{jk}\right]+ \lambda \delta_{ij}\delta_{kl} \\
    \frac{1}{J}\vec{\nabla}(\vec{v}):\left[\frac{\partial \boldsymbol{\tau}}{\partial \boldsymbol{F}} \cdot \boldsymbol{F}^T\right]:\vec{\nabla}(\Delta\vec{u})
    &=\frac{1}{J}\left[\mu \vec{\nabla}(\vec{v}):[\vec{\nabla}(\Delta\vec{u})\cdot\boldsymbol{b} + \boldsymbol{b}\cdot\vec{\nabla}(\Delta\vec{u})^T]
      + \lambda \text{Tr}(\vec{\nabla}(\vec{v}))\text{Tr}(\vec{\nabla}(\Delta\vec{u}))\right]

and the geometric contribution

..  math::
    -
    \frac{1}{J}\vec{\nabla}(\vec{v}):\left[\boldsymbol{\tau}\square\boldsymbol{1}\right]^{T_{R}}:\vec{\nabla}(\Delta\vec{u})
    =-\frac{1}{J}\vec{\nabla}(\vec{v}):[\boldsymbol{\tau}\cdot\vec{\nabla}(\Delta\vec{u})^T]


The gradients of the test function :math:`\vec{\eta}` and the displacement increment :math:`\Delta\vec{u}` are taken
w.r.t. the current configuration. 
The Deformation gradient :math:`\boldsymbol{F}=\boldsymbol{1}+\vec{\nabla}_X(\vec{u})` needs the gradient of the displacement field \:math:`vec{u}` w.r.t. the reference configuration.
To get the gradients w.r.t. the current configuration we need to update the mesh and basis functions in every
Newton iteration. 
However, the deformation gradient is always computed w.r.t. the reference configuration, so we also need to compute
and interpolate these gradients by the total deformation :math:`u` w.r.t. the initial mesh and basis in every iteration.

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
gradbasis = Basis(mesh, element, intorder=1)

mu, lmbda = 1., 2.

@LinearForm
def L(v, w):
    dudX  = w['orig_gradu'] 
    F     = dudX + identity(dudX)
    J     = det(F)
    lnJ   = np.log(J)
    I     = identity(F)
    b     = mul(F,transpose(F))
    tau   = mu * (b - I) + lmbda * lnJ * I
    return 1/J*ddot(tau,grad(v))
@BilinearForm
def a(u, v, w):
    dudX  = w['orig_gradu'] 
    F     = dudX + identity(dudX)
    J     = det(F)
    lnJ   = np.log(J)
    I     = identity(F)
    b     = mul(F,transpose(F))
    tau   = mu * (b - I) + lmbda * lnJ * I

    stiff = 1/J*( mu*ddot( grad(v), mul(grad(u),b) + mul(b,transpose(grad(u))) ) + lmbda*trace(grad(v))*trace(grad(u)) )
    geo   = -1/J*ddot(grad(v), mul(tau, transpose(grad(u))))
    return stiff+geo
# Dirichlet boundary conditions
dofs = basis.get_dofs('right')
dirichlet = basis.get_dofs({'right', 'left'})
right  = -0.1
tol    = 1e-10
nsteps = 7
# Initialization
u = basis.zeros()
x = mesh.p.copy()
X0= mesh.p.copy()
current_mesh = MeshHex(x,mesh.t).with_boundaries(mesh.boundaries)
for step in range(nsteps):
    lam   = (step + 1.) / nsteps # load factor
    theat = lam*np.pi
    for iteration in range(10):
        # update mesh and basis for current configuration
        current_mesh = MeshHex(X0+u[basis.nodal_dofs],current_mesh.t).with_boundaries(mesh.boundaries)
        current_basis= Basis(current_mesh, element, intorder=1)
        # SET DIRICHLET_BC w.r.t. the displacemnts of the reference basis
        du_D = u.copy()
        x, y, z = basis.doflocs[:, dofs.nodal['u^1']]
        du_D[dofs.nodal['u^1']] = lam * right - du_D[dofs.nodal['u^1']]
        du_D[dofs.nodal['u^2']] = (y * np.cos(theat) - z * np.sin(theat) - y- du_D[dofs.nodal['u^2']])
        du_D[dofs.nodal['u^3']] = (y * np.sin(theat) + z * np.cos(theat) - z- du_D[dofs.nodal['u^3']])
        # compute residual and stiffness matrix with current basis but with displacment gradient to reference configuration
        grad_u = gradbasis.interpolate(u).grad
        f = L.assemble(current_basis, orig_gradu=grad_u)
        K = a.assemble(current_basis, orig_gradu=grad_u)
        # solve for newton direction
        du = solve(*condense(K, -f, x=du_D, D=dirichlet))
        norm_du = np.linalg.norm(du)
        u += du
        
        print(1 + iteration, norm_du)
        if norm_du < tol: break

if __name__ == '__main__':
    import vedo
    p = (mesh.translated(u[basis.nodal_dofs])
             .draw('vedo', point_data={'uy': u[basis.nodal_dofs[1]]}))
    uy = u[basis.nodal_dofs[1]]
    p.pointdata.select('uy')
    p.cmap('rainbow', uy, vmin=uy.min(), vmax=uy.max())
    p.properties.SetAmbient(0.85)
    p.properties.SetDiffuse(0.0)
    p.lighting('off')
    edges = p.clone().wireframe(True).color('black').lw(1.5)
    # save image
    plt = vedo.Plotter(offscreen=True)
    plt.add(p, edges)
    plt.reset_camera()  
    plt.screenshot('ex55.png')
    # show interactively
    plt = vedo.Plotter()
    plt.add(p, edges)
    plt.show()


