"""

This redoes the backward-facing step, but as in Gartling (1990), or rather Dyne & Heinrich (1992)

*  Dyne, B. R., & Heinrich, J. C. (1992). Flow over a backward-facing step: a benchmark problem for laminar flow with heat transfer. In B. F. Blackwell & D. W. Pepper (ed.), *Benchmark Problems for Heat Transfer Codes,* Vol. 222 (pp. 73–76). ASME.

* Gartling, D. K. (1990). A test problem for outflow boundary conditions—flow over a backward‐facing step. *International Journal for Numerical Methods in Fluids,* 11, 953–967. doi: 10.1002/fld.1650110704

"""

from skfem import *
from skfem.models.poisson import vector_laplace, laplace, unit_load
from skfem.models.general import divergence, rot

from functools import partial
from itertools import cycle, islice, starmap
from pathlib import Path
from typing import Tuple

from matplotlib.pyplot import subplots
from matplotlib.tri import Triangulation
import numpy as np
from scipy.sparse import bmat, block_diag, csr_matrix

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from pacopy import natural


@linear_form
def acceleration(v, dv, w):
    """Compute the vector (v, u . grad u) for given velocity u."""
    u, du = w.w, w.dw
    # TODO: Handle the indices more cleverly
    return (v[0] * (u[0] * du[0][0] + u[1] * du[0][1])
            + v[1] * (u[0] * du[1][0] + u[1] * du[1][1]))


@bilinear_form
def acceleration_jacobian(u, du, v, dv, w):
    """Compute (v, w . grad u + u . grad w) for given velocity w"""
    return (v[0] * (w.w[0] * du[0][0] + w.w[1] * du[0][1]
                    + u[0] * w.dw[0][0] + u[1] * w.dw[0][1])
            + v[1] * (w.w[0] * du[1][0] + w.w[1] * du[1][1]
                      + u[0] * w.dw[1][0] + u[1] * w.dw[1][1]))


class BackwardFacingStep:

    element = {'u': ElementVectorH1(ElementTriP2()),
               'p': ElementTriP1()}

    def __init__(self,
                 length: float = 30.,
                 lcar: float = .5):

        self.mesh = self.make_mesh(self.make_geom(length, lcar))
        self.basis = {variable: InteriorBasis(self.mesh, e, intorder=3)
                      for variable, e in self.element.items()}
        self.basis['inlet'] = FacetBasis(
            self.mesh, self.element['u'],
            facets=self.mesh.boundaries['inlet'])
        self.basis['V'] = InteriorBasis(self.mesh, ElementTriP2())
        self.D = np.setdiff1d(
            self.basis['u'].get_dofs().all(),
            self.basis['u'].get_dofs(self.mesh.boundaries['outlet']).all())

        A = asm(vector_laplace, self.basis['u'])
        B = asm(divergence, self.basis['u'], self.basis['p'])
        self.S = bmat([[A, -B.T],
                       [-B, None]]).tocsr()
        self.I = np.setdiff1d(np.arange(self.S.shape[0]), self.D)

    def make_geom(self, length: float, lcar: float) -> Geometry:
        # Gartling (1990, figure 1), Dyne & Heinrich (1992, figure 1)
        geom = Geometry()

        points = []
        for point in [[0., 0., 0.],
                      [0., -.5, 0.],
                      [length, -.5, 0.],
                      [length, .5, 0.],
                      [0., .5, 0.]]:
            points.append(geom.add_point(point, lcar))

        lines = list(starmap(geom.add_line,
                             zip(points,
                                 islice(cycle(points), 1, None))))

        list(starmap(geom.add_physical, zip(lines, ['step',
                                                    'floor',
                                                    'outlet',
                                                    'ceiling',
                                                    'inlet'])))

        geom.add_physical(geom.add_plane_surface(geom.add_line_loop(lines)),
                          'domain')

        return geom

    @staticmethod
    def make_mesh(geom: Geometry) -> MeshTri:
            return MeshTri.from_meshio(generate_mesh(geom, dim=2))

    def inlet_dofs(self):
        inlet_dofs_ = self.basis['u'].get_dofs(self.mesh.boundaries['inlet'])
        inlet_dofs_ = self.basis['inlet'].get_dofs(
            self.mesh.boundaries['inlet'])
        return np.concatenate([inlet_dofs_.nodal[f'u^{1}'],
                               inlet_dofs_.facet[f'u^{1}']])

    @staticmethod
    def parabolic(x, y):
        """return the plane Poiseuille parabolic inlet profile

        (Dyne & Heinrich 1992, @ p. 73)
        """
        return ((24 * y * (.5 - y), np.zeros_like(y)))

    def make_vector(self):
        """prepopulate solution vector with Dirichlet conditions"""
        uvp = np.zeros(self.basis['u'].N + self.basis['p'].N)
        I = self.inlet_dofs()
        uvp[I] = L2_projection(self.parabolic, self.basis['inlet'], I)
        return uvp

    def creeping(self):
        """return the solution for zero Reynolds number"""
        uvp = self.make_vector()
        uvp[self.I] = solve(
            *condense(self.S, np.zeros_like(uvp), uvp, self.I))
        return uvp

    def split(self, solution: np.ndarray) -> Tuple[np.ndarray,
                                                   np.ndarray]:
        """return velocity and pressure separately"""
        return np.split(solution, [self.basis['u'].N])

    def streamfunction(self, velocity: np.ndarray) -> np.ndarray:
        A = asm(laplace, self.basis['V'])
        psi = np.zeros(self.basis['V'].N)
        D = np.concatenate(
            [self.basis['V'].get_dofs(self.mesh.boundaries[p]).all()
             for p in ['step', 'floor']])
        I = self.basis['V'].complement_dofs(D)
        vorticity = asm(rot, self.basis['V'],
                              w=[self.basis['V'].interpolate(velocity[i::2])
                                 for i in range(2)])
        psi[I] = solve(*condense(A, vorticity, I=I))
        return psi

    @property
    def termini(self):
        return self.mesh.facets[:, np.concatenate(
            list(self.mesh.boundaries.values()))]

    def mesh_plot(self):
        """return Axes showing boundary of mesh"""
        _, ax = subplots()
        ax.plot(*self.mesh.p[:, self.termini], color='k')
        return ax

    def triangulation(self):
        return Triangulation(
            self.mesh.p[0, :], self.mesh.p[1, :], self.mesh.t.T)

    def streamlines(self, psi: np.ndarray, n: int = 11, ax=None):
        """reproduce figure 2 of Dyne & Heinrich (1990)"""
        if ax is None:
            ax = self.mesh_plot()
        plot = partial(ax.tricontour,
                       self.triangulation(),
                       psi[self.basis['V'].nodal_dofs.flatten()],
                       linewidths=.5)
        for levels, color, style in [
                ([0., .1, .15, .2, .25, .3, .35, .4, .45, .5],
                 'k', ['dashed'] + ['solid']*10 + ['dashed']),
                ([.501, .504], 'r', 'solid'),
                ([-.03, -.02, -.01], 'g', 'solid')]:
            plot(levels=levels, colors=color, linestyles=style)

        ax.set_aspect(1.)
        ax.set_xlim((0., 12.))
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
        return np.hstack([asm(acceleration, self.basis['u'], w=u),
                          np.zeros(self.basis['p'].N)])

    def f(self, uvp: np.ndarray, reynolds: float) -> np.ndarray:
        """Return the residual of a given velocity-pressure solution"""
        out = self.S @ uvp + reynolds * self.N(uvp)

        out[self.D] = uvp[self.D] - self.make_vector()[self.D]
        return out

    def jacobian_solver(self,
                        uvp: np.ndarray,
                        reynolds: float,
                        rhs: np.ndarray) -> np.ndarray:
        duvp = self.make_vector() - uvp
        u = self.basis['u'].interpolate(self.split(uvp)[0])
        duvp[self.I] = solve(*condense(
            self.S + reynolds * block_diag(
                [asm(acceleration_jacobian, self.basis['u'], w=u),
                 csr_matrix((self.basis['p'].N,)*2)]),
            rhs, duvp, I=self.I))
        return duvp


bfs = BackwardFacingStep(lcar=.5**4)


re = [8e2]
velocity = {}

def callback(k, reynolds, uvp):
    print(f'Re = {reynolds}')

    if reynolds in re:
        velocity[reynolds] = bfs.split(uvp)[0]
        

natural(bfs, bfs.make_vector(), 1e2, callback,
        lambda_stepsize0=200.,
        lambda_stepsize_max=200.,
        max_newton_steps=2**3,
        use_first_order_predictor=False,
        milestones=re)

# Forced convection

reynolds = re[-1]
prandtl = 0.7                  # =kinematic viscosity / thermal diffusivity
peclet = prandtl * reynolds
print(f'Re = {reynolds}, Pe = {peclet}')


def fully_developed(x, y):
    """return the fully developed temperature profile"""
    return 1.5 * (1 - (4 * y - 1)**2) * (1 - (4 * y - 1)**2 / 5)


@bilinear_form
def advection(u, du, v, dv, w):
    velocity = w.w
    return v * sum(velocity * du)


cooling = asm(
    unit_load,
    FacetBasis(bfs.mesh, bfs.basis['V'].elem,
               facets=np.concatenate(
                   [bfs.mesh.boundaries[patch] for patch in
                    ['floor', 'ceiling']])))
A = (asm(laplace, bfs.basis['V']) +
     peclet * asm(advection, bfs.basis['V'],
                  w=[bfs.basis['V'].interpolate(velocity[reynolds][i::2])
                     for i in range(2)]))
inlet_basis = FacetBasis(bfs.mesh, bfs.basis['V'].elem,
                         facets=bfs.mesh.boundaries['inlet'])
inlet_dofs = inlet_basis.get_dofs(bfs.mesh.boundaries['inlet']).all()

temperature = np.zeros(bfs.basis['V'].N)
temperature[inlet_dofs] = L2_projection(
    fully_developed, inlet_basis, inlet_dofs)
temperature[bfs.basis['V'].complement_dofs(inlet_dofs)] = solve(*condense(
    A, -48/5*cooling, temperature, D=inlet_dofs))


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    name = splitext(argv[0])[0]

    v = velocity[reynolds]

    fig, axes = subplots(2)
    for ax in axes:
        ax.plot(*bfs.mesh.p[:, bfs.termini], color='k')
    bfs.streamlines(bfs.streamfunction(v), ax=axes[0])
    axes[0].set_title('streamlines')
    
    axes[1].tricontour(         # Dyne & Heinrich (1992, figure 3)
        bfs.triangulation(),
        temperature[bfs.basis['V'].nodal_dofs.flatten()],
        levels=np.linspace(-4., 1., 11))
    axes[1].set_aspect(1.)
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_title('isotherms')
    
    fig.savefig(f'{name}_solution.png',
                bbox_inches="tight", pad_inches=0)
