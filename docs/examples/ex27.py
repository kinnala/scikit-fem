r"""Backward-facing step.

.. warning::
   This example requires the external package `pacopy 0.1.2 <https://pypi.org/project/pacopy/0.1.2/>`_.


In this example, natural parameter continuation is used to extend
the Stokes equations over a backward-facing step to finite Reynolds
number; this means defining a residual for the nonlinear problem and its
derivatives with respect to the solution and to the parameter (here Reynolds
number).

Compared to the `Stokes equations
<https://en.wikipedia.org/wiki/Stokes_flow#Incompressible_flow_of_Newtonian_fluids>`_,
the Navier-Stokes equation has an additional term.  This term appears
multiplied by the Reynolds number if the problem is nondimensionalized using a
characteristic length (the height of the step) and velocity (the average over
the inlet).  Thus, Reynold number serves as a convenient parameter for
numerical continuation.  The nondimensionalized, time-independent Navier-Stokes
equations read

.. math::
   \left\{
   \begin{aligned}
     -\Delta \boldsymbol{u} + \nabla p - \mathrm{Re}\,(\nabla\boldsymbol{u})\boldsymbol{u} &= \boldsymbol{0},\\
     \nabla\cdot\boldsymbol{u} &= 0,
   \end{aligned}
   \right.
where :math:`\boldsymbol{u}` is the velocity field, :math:`p` is the pressure
field and :math:`\mathrm{Re} > 0` is the Reynolds number.  The weak formulation
reads

.. math::
   \begin{aligned}
    &\int_\Omega \nabla\boldsymbol{u} : \nabla\boldsymbol{v}\,\mathrm{d}x - \int_{\Omega} \nabla\cdot\boldsymbol{v} \,p \,\mathrm{d}x
   \\
    &\qquad+ \int_{\Omega} \nabla\cdot\boldsymbol{u} \,q \,\mathrm{d}x - \mathrm{Re} \int_{\Omega} (\nabla\boldsymbol{u})\boldsymbol{u} \cdot \boldsymbol{v}\,\mathrm{d}x = 0,
   \end{aligned}
where :math:`\Omega` is the fluid domain and :math:`(\boldsymbol{v}, q)` are
test functions.
The Jacobian of the last nonlinear term is

.. math::
   -\mathrm{Re} \int_\Omega ((\nabla \delta \boldsymbol{u}) \boldsymbol{u} + (\nabla \boldsymbol{u}) \delta \boldsymbol{u}) \cdot \boldsymbol{v} \,\mathrm{d}x.

"""

from skfem import *
from skfem.helpers import grad, dot
from skfem.models.poisson import vector_laplace, laplace
from skfem.models.general import divergence, rot
import skfem.io.json

from functools import partial
from itertools import cycle, islice
from pathlib import Path
from typing import Tuple, Iterable

from matplotlib.pyplot import subplots
from matplotlib.tri import Triangulation
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import bmat, block_diag, csr_matrix


class NewtonConvergenceError(Exception):
    """

    Copied from pacopy 0.1.2, copyright N. Schlömer & G. D. McBain 2020 under
    the MIT Licence.

    https://raw.githubusercontent.com/nschloe/pacopy/v0.1.2/pacopy/natural.py

    """
    pass


# def newton(f, jacobian_solver, norm2, u0, tol=1.0e-10, max_iter=20, verbose=True):
#     """return a root of f near u0
    
#     Copied from pacopy 0.1.2, copyright N. Schlömer & G. D. McBain 2020 under
#     the MIT Licence.

#     https://raw.githubusercontent.com/nschloe/pacopy/v0.1.2/pacopy/natural.py

#     """
    
#     u = u0

#     fu = f(u)
#     nrm = np.sqrt(norm2(fu))
#     if verbose:
#         print(f"||F(u)|| = {nrm:e}")

#     k = 0
#     while k < max_iter:
#         if nrm < tol:
#             break
#         du = jacobian_solver(u, -fu)
#         u += du
#         fu = f(u)
#         nrm = np.sqrt(norm2(fu))
#         k += 1
#         if verbose:
#             print(f"||F(u)|| = {nrm:e}")

#     is_converged = nrm < tol

#     if not is_converged:
#         raise NewtonConvergenceError(
#             f"Newton's method didn't converge after {k} steps."
#         )

#     return u, k



def natural(
    problem,
    u0,
    lambda0,
    callback,
    lambda_stepsize0=1.0e-1,
    lambda_stepsize_max=float("inf"),
    lambda_stepsize_aggressiveness=2,
    max_newton_steps=5,
    newton_tol=1e-9,
    max_steps=float("inf"),
    verbose=True,
    use_first_order_predictor=True,
    milestones=None,
):
    """Natural parameter continuation.

    The most naive parameter continuation. This simply solves :math:`F(u, \\lambda_0)=0`
    using Newton's method, then changes :math:`\\lambda` slightly and solves again using
    the previous solution as an initial guess. Cannot handle turning points.

    Copied from pacopy 0.1.2, copyright N. Schlömer & G. D. McBain 2020 under
    the MIT Licence.

    https://raw.githubusercontent.com/nschloe/pacopy/v0.1.2/pacopy/natural.py

    Args:
        problem: Instance of the problem class
        u0: Initial guess
        lambda0: Initial parameter value
        callback: Callback function
        lambda_stepsize0 (float): Initial step size
        lambda_stepsize_aggressiveness (float): The step size is adapted after each step
            such that :code:`max_newton_steps` is exhausted approximately. This parameter
            determines how aggressively the the step size is increased if too few Newton
            steps were used.
        lambda_stepsize_max (float): Maximum step size
        max_newton_steps (int): Maxmimum number of Newton steps
        newton_tol (float): Newton tolerance
        max_steps (int): Maximum number of continuation steps
        verbose (bool): Verbose output
        use_first_order_predictor (bool): Once a solution has been found, one can use it
            to bootstrap the Newton process for the next iteration (order 0). Another
            possibility is to use :math:`u - s J^{-1}(u, \\lambda)
            \\frac{df}{d\\lambda}`, a first-order approximation.
        milestones (Optional[Iterable[float]]): Don't step over these values.
    """
    lmbda = lambda0
    if milestones is not None:
        milestones = iter(milestones)

    k = 0
    sol = problem.solve(u0,
                        lmbda,
                        tol=newton_tol,
                        max_iter=max_newton_steps)
    if sol.nit >= max_newton_steps:
        print("No convergence for initial step.")
        raise NewtonConvergenceError
    u = sol.x

    callback(k, lmbda, u)
    k += 1

    lambda_stepsize = lambda_stepsize0
    if milestones is not None:
        milestone = next(milestones)

    while True:
        if k > max_steps:
            break

        if verbose:
            print(
                f"Step {k}: lambda  {lmbda:.3e} + {lambda_stepsize:.3e}  "
                f"->  {lmbda + lambda_stepsize:.3e}"
            )

        # Predictor
        lmbda += lambda_stepsize
        if milestones:
            lmbda = min(lmbda, milestone)
        if use_first_order_predictor:
            du_dlmbda = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))
            u0 = u + du_dlmbda * lambda_stepsize
        else:
            u0 = u

        # Corrector
        sol = problem.solve(u0, lmbda, tol=newton_tol, max_iter=max_newton_steps)
        if sol.nit >= max_newton_steps:
            if verbose:
                print(f"No convergence for lambda={lmbda}.")
            lmbda -= lambda_stepsize
            lambda_stepsize /= 2
            continue
        u = sol.x

        callback(k, lmbda, u)
        k += 1
        if milestones is not None and lmbda == milestone:
            try:
                milestone = next(milestones)
            except StopIteration:
                break
        else:
            lambda_stepsize *= (
                1
                + lambda_stepsize_aggressiveness
                * ((max_newton_steps - sol.nit) / (max_newton_steps - 1)) ** 2
            )
            lambda_stepsize = min(lambda_stepsize, lambda_stepsize_max)


@LinearForm
def acceleration(v, w):
    """Compute the vector (v, u . grad u) for given velocity u

    passed in via `wind` after having been interpolated onto its
    quadrature points.

    In Cartesian tensorial indicial notation, the integrand is

    .. math::

        u_j u_{i,j} v_i.

    """
    return np.einsum('j...,ij...,i...', w['wind'], grad(w['wind']), v)


@BilinearForm
def acceleration_jacobian(u, v, w):
    """Compute (v, w . grad u + u . grad w) for given velocity w

    passed in via w after having been interpolated onto its quadrature
    points.

    In Cartesian tensorial indicial notation, the integrand is

    .. math::

       (w_j du_{i,j} + u_j dw_{i,j}) v_i

    """
    return dot(np.einsum('j...,ij...->i...', w['wind'], grad(u))
               + np.einsum('j...,ij...->i...', u, grad(w['wind'])), v)


class BackwardFacingStep:

    element = {'u': ElementVectorH1(ElementTriP2()),
               'p': ElementTriP1()}

    def __init__(self,
                 length: float = 35.):

        self.mesh = skfem.io.json.from_file(Path(__file__).with_suffix(".json"))
        self.basis = {variable: InteriorBasis(self.mesh, e, intorder=3)
                      for variable, e in self.element.items()}
        self.basis['inlet'] = FacetBasis(self.mesh, self.element['u'],
                                         facets=self.mesh.boundaries['inlet'])
        self.basis['psi'] = InteriorBasis(self.mesh, ElementTriP2())
        self.D = np.concatenate([b.all() for b in self.basis['u'].find_dofs().values()])

        A = asm(vector_laplace, self.basis['u'])
        B = asm(divergence, self.basis['u'], self.basis['p'])
        self.S = bmat([[A, -B.T],
                       [-B, None]], 'csr')
        self.I = np.setdiff1d(np.arange(self.S.shape[0]), self.D)

    def inlet_dofs(self):
        return self.basis['inlet'].find_dofs()['inlet'].all()

    @staticmethod
    def parabolic(x, y):
        """return the plane Poiseuille parabolic inlet profile"""
        return 4 * y * (1. - y), np.zeros_like(y)

    def make_vector(self):
        """prepopulate solution vector with Dirichlet conditions"""
        uvp = np.zeros(self.basis['u'].N + self.basis['p'].N)
        I = self.inlet_dofs()
        uvp[I] = L2_projection(self.parabolic, self.basis['inlet'], I)
        return uvp

    def split(self, solution: np.ndarray) -> Tuple[np.ndarray,
                                                   np.ndarray]:
        """return velocity and pressure separately"""
        return np.split(solution, [self.basis['u'].N])

    def streamfunction(self, velocity: np.ndarray) -> np.ndarray:
        A = asm(laplace, self.basis['psi'])
        psi = np.zeros(self.basis['psi'].N)
        vorticity = asm(rot, self.basis['psi'],
                        w=[self.basis['psi'].interpolate(velocity[i::2])
                           for i in range(2)])
        psi = solve(*condense(A, vorticity, D=self.basis['psi'].find_dofs()['floor'].all()))
        return psi

    def mesh_plot(self):
        """return Axes showing boundary of mesh"""
        termini = self.mesh.facets[:, np.concatenate(list(
            self.mesh.boundaries.values()))]
        _, ax = subplots()
        ax.plot(*self.mesh.p[:, termini], color='k')
        return ax

    def triangulation(self):
        return Triangulation(*self.mesh.p, self.mesh.t.T)

    def streamlines(self, psi: np.ndarray, n: int = 11, ax=None):
        if ax is None:
            ax = self.mesh_plot()
        n_streamlines = n
        plot = partial(ax.tricontour,
                       self.triangulation(),
                       psi[self.basis['psi'].nodal_dofs.flatten()],
                       linewidths=.5)
        for levels, color, style in [
            (np.linspace(0, 2/3, n_streamlines),
             'k',
             ['dashed'] + ['solid']*(n_streamlines - 2) + ['dashed']),
            (np.linspace(2/3, max(psi), n_streamlines)[0:],
             'r', 'solid'),
            (np.linspace(min(psi), 0, n_streamlines)[:-1],
             'g', 'solid')]:
            plot(levels=levels, colors=color, linestyles=style)

        ax.set_aspect(1.)
        ax.axis('off')
        return ax

    def inner(self, u: np.ndarray,  v: np.ndarray) -> float:
        """return the inner product of two solutions

        using just the velocity, ignoring the pressure

        """
        return self.split(u)[0] @ self.split(v)[0]

    def norm2_r(self, u: np.ndarray) -> float:
        return self.inner(u, u)

    def N(self, uvp: np.ndarray) -> np.ndarray:
        u = self.basis['u'].interpolate(self.split(uvp)[0])
        return np.hstack([asm(acceleration, self.basis['u'], wind=u),
                          np.zeros(self.basis['p'].N)])

    def f(self, uvp: np.ndarray, reynolds: float) -> np.ndarray:
        """Return the residual of a given velocity-pressure solution"""
        out = self.S @ uvp + reynolds * self.N(uvp)

        out[self.D] = uvp[self.D] - self.make_vector()[self.D]
        return out

    def solve(self,
              uvp: np.ndarray,
              reynolds: float,
              tol: float,
              max_iter: int) -> OptimizeResult:
        """return the solution at the given Reynolds number

        Based on newton from pacopy 0.1.2 (copyright nschloe, gmcbain;
        MIT Licence).

        """

        u = uvp
        fu = self.f(u, reynolds)
        nfev = 1
        nrm = np.sqrt(self.norm2_r(fu))
        print(f"||F(u)|| = {nrm:e}")
        
        njev = k = 0
        while k < max_iter:
            if nrm < tol:
                return OptimizeResult(x=u, 
                                      success=True,
                                      nfev=nfev,
                                      njev=njev,
                                      nit=k)
            du = self.jacobian_solver(u, reynolds, -fu)
            njev += 1
            u += du
            fu = self.f(u, reynolds)
            nfev += 1
            nrm = np.sqrt(self.norm2_r(fu))
            k += 1
            print(f"||F(u)|| = {nrm:e}")
        
        return OptimizeResult(x=u, 
                              success=False, 
                              message="Newton's method didn't converge.", 
                              nit=k,
                              nfev=nfev,
                              njev=njev)
        
    def df_dlmbda(self, uvp: np.ndarray, reynolds: float) -> np.ndarray:
        out = self.N(uvp)
        out[self.D] = 0.
        return out

    def jacobian_solver(self,
                        uvp: np.ndarray,
                        reynolds: float,
                        rhs: np.ndarray) -> np.ndarray:
        u = self.basis['u'].interpolate(self.split(uvp)[0])
        duvp = solve(*condense(
            self.S +
            reynolds
            * block_diag([asm(acceleration_jacobian, self.basis['u'], wind=u),
                          csr_matrix((self.basis['p'].N,)*2)]),
            rhs, self.make_vector() - uvp, I=self.I))
        return duvp


bfs = BackwardFacingStep()
psi = {}


def callback(_: int,
             reynolds: float,
             uvp: np.ndarray,
             milestones=Iterable[float]):
    """Echo the Reynolds number and plot streamlines at milestones"""
    print(f'Re = {reynolds}')

    if reynolds in milestones:

        psi[reynolds] = bfs.streamfunction(bfs.split(uvp)[0])

        if __name__ == '__main__':
            ax = bfs.streamlines(psi[reynolds])
            ax.set_title(f'Re = {reynolds}')
            ax.get_figure().savefig(Path(__file__).with_name(
                f'{Path(__file__).stem}-{reynolds}-psi.png'),
                                    bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':
    milestones = [50., 150., 450., 750.]
else:
    milestones = [50.]

natural(bfs, bfs.make_vector(), 0.,
        partial(callback,
                milestones=milestones),
        lambda_stepsize0=50.,
        milestones=milestones)
