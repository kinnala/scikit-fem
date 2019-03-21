from itertools import cycle, islice
from typing import Tuple

from matplotlib.pyplot import subplots
import numpy as np
from scipy.sparse import bmat, block_diag, csr_matrix

import meshio
from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem import *
from skfem.models.poisson import vector_laplace, laplace
from skfem.models.general import divergence, rot

from pacopy import natural


@linear_form
def acceleration(v, dv, w):
    """Compute the vector (v, u . grad u) for given velocity u."""
    u, du = w.w, w.dw
    # TODO: Handle the indices more cleverly
    return (v[0] * (u[0] * du[0][0] + u[1] * du[1][0])
            + v[1] * (u[0] * du[0][1] + u[1] * du[1][1]))


@bilinear_form
def acceleration_jacobian(u, du, v, dv, w):
    """Compute (v, w . grad u + u . grad w) for given velocity w"""
    return (v[0] * (w.w[0] * du[0][0] + w.w[1] * du[1][0]
                    + u[0] * w.dw[0][0] + u[1] * w.dw[1][0])
            + v[1] * (w.w[0] * du[0][1] + w.w[1] * du[1][1]
                      + u[0] * w.dw[0][1] + u[1] * w.dw[1][1]))


class BackwardFacingStep:

    element = {'u': ElementVectorH1(ElementTriP2()),
               'p': ElementTriP1()}

    def __init__(self,
                 length: float = 35.,
                 lcar: float = 1.):

        self.mesh = self.make_mesh(self.make_geom(length, lcar))
        self.basis = {variable: InteriorBasis(self.mesh, e, intorder=3)
                      for variable, e in self.element.items()}
        self.basis['inlet'] = FacetBasis(self.mesh, self.element['u'],
                                         facets=self.mesh.boundaries['inlet'])
        self.basis['psi'] = InteriorBasis(self.mesh, ElementTriP2())
        self.D = np.setdiff1d(
            self.basis['u'].get_dofs().all(),
            self.basis['u'].get_dofs(self.mesh.boundaries['outlet']).all())

        A = asm(vector_laplace, self.basis['u'])
        B = asm(divergence, self.basis['u'], self.basis['p'])
        self.S = bmat([[A, -B.T],
                       [-B, None]]).tocsr()
        self.I = np.setdiff1d(np.arange(self.S.shape[0]), self.D)
        
    def make_geom(self, length: float, lcar: float) -> Geometry:
        # Barkley et al (2002, figure 3 a - c)
        geom = Geometry()

        points = []
        for point in [[0, -1, 0],
                      [length, -1, 0],
                      [length, 1, 0],
                      [-1, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 0]]:
            points.append(geom.add_point(point, lcar))

        lines = []
        for termini in zip(points,
                           islice(cycle(points), 1, None)):
            lines.append(geom.add_line(*termini))

        for k, label in [([1], 'outlet'),
                         ([2], 'ceiling'),
                         ([3], 'inlet'),
                         ([0, 4, 5], 'floor')]:
            geom.add_physical_line(list(np.array(lines)[k]), label)

        geom.add_physical_surface(
            geom.add_plane_surface(geom.add_line_loop(lines)), 'domain')

        return geom

    @staticmethod
    def make_mesh(geom: Geometry) -> MeshTri:
            return MeshTri.from_meshio(meshio.Mesh(*generate_mesh(geom, dim=2)))

    def inlet_dofs(self):
        inlet_dofs_ = self.basis['u'].get_dofs(self.mesh.boundaries['inlet'])
        inlet_dofs_ = self.basis['inlet'].get_dofs(
            self.mesh.boundaries['inlet'])
        return np.concatenate([inlet_dofs_.nodal[f'u^{1}'],
                               inlet_dofs_.facet[f'u^{1}']])

    @staticmethod
    def parabolic(x, y):
        """return the plane Poiseuille parabolic inlet profile"""
        return ((4 * y * (1. - y), np.zeros_like(y)))

    def make_vector(self):
        """prepopulate solution vector with Dirichlet conditions"""
        uvp = np.zeros(self.basis['u'].N + self.basis['p'].N)
        I = self.inlet_dofs()
        uvp[I] = L2_projection(self.parabolic, self.basis['inlet'], I)
        return uvp

    def creeping(self):
        """return the solution for zero Reynolds number"""
        uvp = self.make_vector()
        uvp[self.I] = solve(*condense(self.S, np.zeros_like(uvp), uvp, self.I))
        return uvp

    def split(self, solution: np.ndarray) -> Tuple[np.ndarray,
                                                   np.ndarray]:
        """return velocity and pressure separately"""
        return np.split(solution, [self.basis['u'].N])

    def streamfunction(self, velocity: np.ndarray) -> np.ndarray:
        A = asm(laplace, self.basis['psi'])
        psi = np.zeros(self.basis['psi'].N)
        D = self.basis['psi'].get_dofs(self.mesh.boundaries['floor']).all()
        I = self.basis['psi'].complement_dofs(D)
        vorticity = asm(rot, self.basis['psi'],
                        w=[self.basis['psi'].interpolate(velocity[i::2])
                           for i in range(2)])
        psi[I] = solve(*condense(A, vorticity, I=I))
        return psi

    def mesh_plot(self):
        termini = self.mesh.facets[:, np.concatenate(list(
            self.mesh.boundaries.values()))]
        _, ax = subplots()
        ax.plot(*self.mesh.p[:, termini], color='k')
        return ax

    def triangulation(self):
        return Triangulation(
            self.mesh.p[0, :], self.mesh.p[1, :], self.mesh.t.T)

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
        return np.hstack([asm(acceleration, self.basis['u'], w=u),
                          np.zeros(self.basis['p'].N)])

    def f(self, uvp: np.ndarray, reynolds: float) -> np.ndarray:
        """Return the residual of a given velocity-pressure solution"""
        out = self.S @ uvp - reynolds * self.N(uvp)

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
        duvp = np.zeros_like(uvp)
        u = self.basis['u'].interpolate(self.split(uvp)[0])
        duvp[self.I] = solve(*condense(
            self.S -
            reynolds
            * block_diag([asm(acceleration_jacobian, self.basis['u'], w=u),
                          csr_matrix((self.basis['p'].N,)*2)]),
            rhs, I=self.I))
        return duvp


bfs = BackwardFacingStep(lcar=.5**3)


class RangeException(Exception):
    pass


re = []
ax = bfs.mesh_plot()

def callback(k, reynolds, uvp):
    print(f'Re = {reynolds}')
    re.append(reynolds)

    ax.cla()
    bfs.streamlines(bfs.streamfunction(bfs.split(uvp)[0]), ax=ax)
    ax.set_title(f'Re = {reynolds}')
    ax.get_figure().savefig('psi.png', bbox_inches="tight", pad_inches=0)
    
    if reynolds > 750.:
        raise RangeException


if __name__ == '__main__':

    from functools import partial
    from os.path import splitext
    from sys import argv

    from matplotlib.tri import Triangulation

    name = splitext(argv[0])[0]


    uvp0 = bfs.creeping()
    u0, p0 = bfs.split(uvp0)
    bfs.mesh.plot(p0)
    bfs.mesh.savefig(f'{name}_p0.png', bbox_inches="tight", pad_inches=0)

    psi0 = bfs.streamfunction(u0)
    ax_psi = bfs.streamlines(psi0)
    ax_psi.get_figure().savefig(f'{name}_psi0.png',
                                bbox_inches="tight", pad_inches=0)
                                
    try:
        natural(bfs, uvp0, 0., callback, lambda_stepsize_max=150.)
    except RangeException:
        print('Re = ', re)
        print('Left range')

    # mesh.plot(pressure)
    # mesh.savefig(f'{name}-pressure.png', bbox_inches='tight', pad_inches=0)

    # bfs.mesh.save(f'{name}-velocity.vtk',
    #               velocity[bfs.basis['u'].nodal_dofs].T)

    # ax = bfs.streamlines(psi)
    # ax.get_figure().savefig(f'{name}-stream-function.png',
    #                         bbox_inches='tight', pad_inches=0)
