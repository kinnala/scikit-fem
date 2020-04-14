from typing import NamedTuple, Optional

from skfem import *
from skfem.io.meshio import from_meshio
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence

import numpy as np
from scipy.sparse import bmat, spmatrix
from scipy.sparse.linalg import LinearOperator, minres

from pygmsh import generate_mesh
from pygmsh.opencascade import Geometry

try:
    import pyamgcl
    from scipy.sparse.linalg import aslinearoperator

    def build_pc_amg(A: spmatrix, **kwargs) -> LinearOperator:
        """AMG preconditioner"""
        try:
            from pyamgcl import amgcl  # v. 1.3.99+
        except ImportError:
            from pyamgl import amg as ambcl
        return aslinearoperator(amgcl(A, **kwargs))

except ImportError:
    from pyamg import smoothed_aggregation_solver

    def build_pc_amg(A: spmatrix, **kwargs) -> LinearOperator:
        return smoothed_aggregation_solver(A, **kwargs).aspreconditioner()


class Ellipsoid(NamedTuple):

    a: float
    b: float
    c: float

    @property
    def semiaxes(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c])

    def geom(self, lcar: float = .1) -> Geometry:
        geom = Geometry()
        geom.add_ellipsoid([0.]*3, self.semiaxes, lcar)
        return geom

    def mesh(self, geom: Optional[Geometry] = None, **kwargs) -> MeshTet:
        return from_meshio(generate_mesh(geom or self.geom(**kwargs)))

    def pressure(self, x, y, z) -> np.ndarray:
        a, b, c = self.semiaxes
        return (a**2 * (3 * a**2 + b**2) * x * y 
                / (3 * a**4 + 2 * a**2 * b**2 + 3 * b**4))

    def pressure_error(self):

        def form(v, w):
            return v * (w.w - self.pressure(*w.x))

        return LinearForm(form)

ellipsoid = Ellipsoid(.5, .3, .2)
mesh = ellipsoid.mesh()

element = {'u': ElementVectorH1(ElementTetP2()),
           'p': ElementTetP1()}
basis = {variable: InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}


@LinearForm
def body_force(v, w):
    return w.x[0] * v.value[1]


A = asm(vector_laplace, basis['u'])
B = -asm(divergence, basis['u'], basis['p'])
Q = asm(mass, basis['p'])

K = bmat([[A, B.T],
          [B, None]], 'csr')    # no pressure block required for minres

f = np.concatenate([asm(body_force, basis['u']),
                    np.zeros(B.shape[0])])

D = basis['u'].find_dofs()
Kint, fint, u, I = condense(K, f, D=D)
Aint = Kint[:-(basis['p'].N), :-(basis['p'].N)]

Apc = build_pc_amg(Aint)
diagQ = Q.diagonal()


def precondition(uvp):
    uv, p = np.split(uvp, [Aint.shape[0]])
    return np.concatenate([Apc @ uv, p / diagQ])


M = LinearOperator(Kint.shape, precondition, dtype=Q.dtype)

velocity, pressure = np.split(
    solve(Kint, fint, u, I,
          solver=solver_iter_krylov(minres, verbose=True, M=M)),
    [basis['u'].N])




l2error_p = asm(ellipsoid.pressure_error(), basis['p'],
                w=basis['p'].interpolate(pressure))


if __name__ == '__main__':

    from pathlib import Path

    print('L2 error in pressure:', np.sqrt(l2error_p.T @ Q @ l2error_p))
    
    mesh.save(Path(__file__).with_suffix('.vtk'),
              {'velocity': velocity[basis['u'].nodal_dofs].T,
               'pressure': pressure})
