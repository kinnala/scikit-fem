r"""Incompressible Hyperelasticity

This example solves the governing equations describing the mechanical 
response of a nearly incompressible elastomer using a mixed formulation. 
The elastomer, assumed to be made up of a Neo-Hookean solid, occupies the domain 
:math:`\Omega` in the undeformed configuration, with the internal stored energy 
given by 
.. math::
   \int_\Omega\Psi(\mathbf{F})d\mathbf{X};\qquad\Psi(\mathbf{F}) = \mu/2 (I_1 - 3) - \mu \ln(J) + \lambda/2(J-1)^2

where :math:`I_1 = \mathbf{F}:\mathbf{F} = tr(\mathbf{F}^T\mathbf{F})` and
:math:`J = \text{det}(\mathbf{F})` and :math:`\mathbf{F} = \mathbf{I} +
\nabla\mathbf{u}` is the deformation gradient tensor.  The standard variational
formulation in the absence of body force and surface traction can be written as
.. math::
   \min_{u\in\mathcal{K}} \int_\Omega \Psi(\mathbf{F})d\mathbf{X}

where :math:`\mathcal{K}` is a set of kinematically admissible fields that
satisfy the Dirichlet boundary condition(s).  However, this becomes ill-posed
when :math:`\lambda/\mu\rightarrow +\infty`. In order to circumvent this issue,
we consider a mixed variational formulation, namely

.. math::
   \min_{\mathbf{u}\in\mathcal{K}}\max_{p}\int_\Omega \Psi*(\mathbf{F}, p)d\mathbf{X}

where 

.. math::
   \Psi*(\mathbf{F}, p) = p (J-J*) + mu/2(I_1-3) - \mu\ln(J*) + \lambda/2(J*-1)^2

and :math:`J* = (\lambda + p + \sqrt{(\lambda + p)^2 + 4\lambda\mu)}/(2\lambda)`.
The numerical solution to the above problem requires choosing stable finite
element spaces for the displacement (:math:`\mathbf{u}`) and pressure
(:math:`p`). The corresponding weak form is given by

find :math:`(\mathbf{u},p)\in (V_1 x V_2)` such that 

.. math::
   \mathcal{F}_1 = \int_\Omega \left( \mu\mathbf{F} + p\mathbf{F}^{-T} \right) : \nabla\mathbf{v}d\mathbf{X} = 0

and 

.. math::
   \mathcal{F}_2 = \int_\Omega \frac{\partial\Psi*}{\partial p}q\ d\mathbf{X} = 0

for all :math:`(\mathbf{v},q)\in (V_1 x V_2)` and 

.. math::
   V_1 = \left\{\mathbf{u}\ni \mathbf{u}\in (H^1(\Omega))^3 \cap u\in\mathcal{K} \right\}

and

.. math::
   V_2 = \left\{ p\ni p\in L^2(\Omega) \right\}

Here, inspired by it's counterpart in fluid mechanics, we choose the lowest
order Taylor-Hood element (:math:`\mathbb{P}_2-P_1`) which satifies the
Babuska-Brezzi condition. Fore more details on the derivation, see
http://pamies.cee.illinois.edu/Publications_files/IJNME_2015.pdf#page=4&zoom=100,312,414
The weak forms above result in a system of nonlinear algebraic equations for
the degrees of freedom , and therefore needs to be solved using a nonlinear
solver. In the example below, we linearize :math:`\mathcal{F}_1` and
:math:`\mathcal{F}_2` and setup solve for incremental displacement and pressure
dofs.

The following demonstrates uniaxial tension in one direction, and the lateral
edges allowed to remain free. The geometry is a homogeneous unit cube made up
of a Neo-Hookean solid with :math:`\lambda/\mu = 10000`. For this loading and
geometry, in the limit of :math:`\lambda/\mu\rightarrow +\infty`, the
deformation gradient would be given by :math:`\mathbf{F} =
\text{diag}(\lambda,1/\sqrt{\lambda})` and the pressure field admits a closed
form solution :math:`p=-\mu/\ell` where :math:`\ell` is the applied stretch.

As another check, we can also compute the final volume of the deformed solid which,
for a nearly incompressible solid, should be close to the initial undeformed volume.

"""
import numpy as np
from scipy.sparse import bmat
from skfem.helpers import grad, transpose, det, inv, identity
from skfem import *


mu, lmbda = 1., 1.e4


def F1(w):
    u = w["disp"]
    p = w["press"]
    F = grad(u) + identity(u)
    J = det(F)
    Finv = inv(F)
    return p * J * transpose(Finv) + mu * F


def F2(w):
    u = w["disp"]
    p = w["press"].value
    F = grad(u) + identity(u)
    J = det(F)
    Js = .5 * (lmbda + p + 2. * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2)) / lmbda
    dJsdp = ((.25 * lmbda + .25 * p + .5 * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2))
             / (lmbda * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2)))
    return J - (Js + (p + mu / Js - lmbda * (Js - 1)) * dJsdp)


def A11(w):
    u = w["disp"]
    p = w["press"]
    eye = identity(u)
    F = grad(u) + eye
    J = det(F)
    Finv = inv(F)
    L = (p * J * np.einsum("lk...,ji...->ijkl...", Finv, Finv)
         - p * J * np.einsum("jk...,li...->ijkl...", Finv, Finv)
         + mu * np.einsum("ik...,jl...->ijkl...", eye, eye))
    return L


def A12(w):
    u = w["disp"]
    F = grad(u) + identity(u)
    J = det(F)
    Finv = inv(F)
    return J * transpose(Finv)


def A22(w):
    u = w["disp"]
    p = w["press"].value
    Js = .5 * (lmbda + p + 2. * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2)) / lmbda
    dJsdp = ((.25 * lmbda + .25 * p + .5 * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2))
             / (lmbda * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2)))
    d2Jdp2 = .25 * mu / (lmbda * mu + .25 * (lmbda + p) ** 2) ** (3/2)
    L = (-2. * dJsdp - p * d2Jdp2 + mu / Js ** 2 * dJsdp ** 2 - mu / Js * d2Jdp2
         + lmbda * (Js - 1.) * d2Jdp2 + lmbda * dJsdp ** 2)
    return L


def volume(w):
    dw = w["disp"].grad
    F = dw + identity(dw)
    J = det(F)
    return J


mesh = MeshTet().refined(2)
uelem = ElementVectorH1(ElementTetP2())
pelem = ElementTetP1()
elems = {
    "u": uelem,
    "p": pelem
}
basis = {
    field: Basis(mesh, e, intorder=2)
    for field, e in elems.items()
}

du = basis["u"].zeros()
dp = basis["p"].zeros()
stretch_ = 1.

ddofs = [
    basis["u"].find_dofs(
        {"left": mesh.facets_satisfying(lambda x: x[0] < 1.e-6)},
        skip=["u^2", "u^3"]
    ),
    basis["u"].find_dofs(
        {"bottom": mesh.facets_satisfying(lambda x: x[1] < 1.e-6)},
        skip=["u^1", "u^3"]
    ),
    basis["u"].find_dofs(
        {"back": mesh.facets_satisfying(lambda x: x[2] < 1.e-6)},
        skip=["u^1", "u^2"]
    ),
    basis["u"].find_dofs(
        {"front": mesh.facets_satisfying(lambda x: np.abs(x[2] - 1.) < 1e-6)},
        skip=["u^1", "u^2"]
    )
]

dofs = {}
for dof in ddofs:
    dofs.update(dof)

du[dofs["left"].all()] = 0.
du[dofs["bottom"].all()] = 0.
du[dofs["back"].all()] = 0.
du[dofs["front"].all()] = stretch_

I = np.hstack((
    basis["u"].complement_dofs(dofs),
    basis["u"].N + np.arange(basis["p"].N)
))


@LinearForm
def a1(v, w):
    return np.einsum("ij...,ij...", F1(w), grad(v))


@LinearForm
def a2(v, w):
    return F2(w) * v


@BilinearForm
def b11(u, v, w):
    return np.einsum("ijkl...,ij...,kl...", A11(w), grad(u), grad(v))


@BilinearForm
def b12(u, v, w):
    return np.einsum("ij...,ij...", A12(w), grad(v)) * u


@BilinearForm
def b22(u, v, w):
    return A22(w) * u * v


@Functional
def vol(w):
    return volume(w)


for itr in range(12):
    uv = basis["u"].interpolate(du)
    pv = basis["p"].interpolate(dp) 

    K11 = asm(b11, basis["u"], basis["u"], disp=uv, press=pv)
    K12 = asm(b12, basis["p"], basis["u"], disp=uv, press=pv)
    K22 = asm(b22, basis["p"], basis["p"], disp=uv, press=pv)
    f = np.concatenate((
        asm(a1, basis["u"], disp=uv, press=pv),
        asm(a2, basis["p"], disp=uv, press=pv)
    ))
    K = bmat(
        [[K11, K12],
         [K12.T, K22]], "csr"
    )
    uvp = solve(*condense(K, -f, I=I), use_umfpack=True)
    delu, delp = np.split(uvp, [du.shape[0]])
    du += delu
    dp += delp
    normu = np.linalg.norm(delu)
    normp = np.linalg.norm(delp)
    norm_res = np.linalg.norm(f[I])
    print(f"{itr+1}, norm_du: {normu}, norm_dp: {normp}, norm_res: {norm_res}")
    if normu < 1.e-8 and normp < 1.e-8 and norm_res < 1.e-8:
        break


volume_deformed = vol.assemble(basis["u"], disp=basis["u"].interpolate(du))


if __name__ == "__main__":
    mesh.save(
        "example36_results.xdmf",
        {"u": du[basis["u"].nodal_dofs].T, "p": dp[basis["p"].nodal_dofs[0]]},
    )
