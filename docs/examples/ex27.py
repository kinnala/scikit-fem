r"""Backward-facing step.

.. warning::
   This example requires the external package `pacopy 0.1.2 <https://pypi.org/project/pacopy/0.1.2/>`_.


In this example, `pacopy 0.1.2 <https://pypi.org/project/pacopy/0.1.2/>`_ is used to extend
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
from pathlib import Path
from typing import Tuple, Iterable

from matplotlib.pyplot import subplots
from matplotlib.tri import Triangulation
import numpy as np
from scipy.sparse import bmat, block_diag, csr_matrix

from pacopy import natural


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

        self.mesh = skfem.io.json.from_file(Path(__file__).parent / 'meshes' / 'backward-facing_step.json')
        self.basis = {variable: Basis(self.mesh, e, intorder=3)
                      for variable, e in self.element.items()}
        self.basis['inlet'] = FacetBasis(self.mesh, self.element['u'],
                                         facets=self.mesh.boundaries['inlet'])
        self.basis['psi'] = self.basis['u'].with_element(ElementTriP2())
        self.D = self.basis['u'].get_dofs([b for b in self.mesh.boundaries])

        A = asm(vector_laplace, self.basis['u'])
        B = asm(divergence, self.basis['u'], self.basis['p'])
        self.S = bmat([[A, -B.T],
                       [-B, None]], 'csr')
        self.I = np.setdiff1d(np.arange(self.S.shape[0]), self.D)

    def inlet_dofs(self):
        return self.basis["inlet"].get_dofs("inlet")

    @staticmethod
    def parabolic(x):
        """return the plane Poiseuille parabolic inlet profile"""
        return np.stack([4 * x[1] * (1. - x[1]), np.zeros_like(x[0])])

    def make_vector(self):
        """prepopulate solution vector with Dirichlet conditions"""
        uvp = np.zeros(self.basis['u'].N + self.basis['p'].N)
        I = self.inlet_dofs()
        uvp[I] = projection(self.parabolic, self.basis['inlet'], I=I)
        return uvp

    def split(self, solution: np.ndarray) -> Tuple[np.ndarray,
                                                   np.ndarray]:
        """return velocity and pressure separately"""
        return np.split(solution, [self.basis['u'].N])

    def streamfunction(self, velocity: np.ndarray) -> np.ndarray:
        A = asm(laplace, self.basis['psi'])
        psi = self.basis['psi'].zeros()
        vorticity = asm(rot, self.basis['psi'],
                        w=self.basis['u'].interpolate(velocity))
        psi = solve(*condense(A, vorticity, D=self.basis['psi'].get_dofs('floor')))
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
                          self.basis['p'].zeros()])

    def f(self, uvp: np.ndarray, reynolds: float) -> np.ndarray:
        """Return the residual of a given velocity-pressure solution"""
        out = self.S @ uvp + reynolds * self.N(uvp)

        out[self.D] = uvp[self.D] - self.make_vector()[self.D]
        return out

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
