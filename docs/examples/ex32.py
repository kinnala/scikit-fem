from skfem import *
from skfem.io.meshio import from_meshio
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import LinearOperator, minres

from pyamg import smoothed_aggregation_solver

from pygmsh import generate_mesh
from pygmsh.opencascade import Geometry

geom = Geometry()
geom.add_ellipsoid([0.]*3, [1., .6, .4], .05)
mesh = from_meshio(generate_mesh(geom))

element = {'u': ElementVectorH1(ElementTetP2()),
           'p': ElementTetP1()}
basis = {variable: InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}


@linear_form
def body_force(v, dv, w):
    return w.x[0] * v[1]


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
Aml = smoothed_aggregation_solver(Aint)

Apc = Aml.aspreconditioner()
diagQ = Q.diagonal()


def precondition(uvp):
    uv, p = np.split(uvp, [Aint.shape[0]])
    return np.concatenate([Apc @ uv, p / diagQ])


M = LinearOperator(Kint.shape, precondition, dtype=Q.dtype)

velocity, pressure = np.split(
    solve(Kint, fint, u, I,
          solver=solver_iter_krylov(minres, verbose=True, M=M)),
    [basis['u'].N])


if __name__ == '__main__':

    from pathlib import Path

    mesh.save(Path(__file__).with_suffix('.vtk'),
              {'velocity': velocity[basis['u'].nodal_dofs].T,
               'pressure': pressure})
