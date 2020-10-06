r"""Block diagonally preconditioned Stokes solver.

.. note::

   This examples requires the external package `pygmsh <https://pypi.org/project/pygmsh/>`_ and an implementation of AMG (either `pyamgcl    <https://pypi.org/project/pyamgcl>`_ or `pyamg <https://pypi.org/project/pyamg/>`_).

This example again solves the Stokes problem,

.. math::
    0 = -\rho^{-1}\nabla p + x\boldsymbol{j} + \nu\Delta\boldsymbol{u}
.. math::
    \nabla\cdot\boldsymbol{u} = 0.

but this time in three dimensions, with an algorithm that scales to reasonably
fine meshes (a million tetrahedra in a few minutes).

The exact solution for this equation in an ellipsoid is known [MCBAIN]_.

With Taylor-Hood elements, the discrete form of the problem is

.. math::
   \begin{matrix}
   A & B.T \\
   B & 0
   \end{matrix}
   \begin{Bmatrix}
   u \\ package\end{Bmatrix}
   =
   \begin{bmatrix}
   x\boldsymbol{j} \\ 0
   \end{bmatrix}

A simple but effective preconditioning strategy [ELMAN]_ is a block-diagonal
approach with algebraic multigrid for the momentum block and a diagonal
approximation to its mass matrix for the pressure.  Algebraic multigrid is
easily accessible from scikit-fem via the external packages AMGCL or PyAMG;
either will do here, and in either case the viscous matrix A is condensed to
eliminate the essential boundary conditions (here zero velocity on the walls)
and then passed to the external AMG library.

Because the two parts of the preconditioner differ in form, it is easier to
define their action by a function, wrapped up as a LinearOperator which can
then be passed to the MINRES sparse iterative solver from SciPy.

.. [ELMAN] Elman, H. C., Silvester, D. J.,, Wathen, A. J. (2014). *Finite Elements and Fast Iterative Solvers : with Applications in Incompressible Fluid Dynamics*, ch. 4 'Solution of discrete Stokes problems'.  Oxford University Press.  `doi:10.1093/acprof:oso/9780199678792.001.0001 <https://doi.org/10.1093%2facprof:oso%2f9780199678792.001.0001>`_

.. [McBAIN] McBain, G. D. (2016). `Creeping convection in a horizontally heated ellipsoid <http://people.eng.unimelb.edu.au/imarusic/proceedings/20/548/%20Paper.pdf>`_. *Proceedings of the Twentieth Australasian Fluid Mechanics Conference*.

License
-------

Copyright 2018-2020 scikit-fem developers

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
from typing import NamedTuple, Optional
from packaging import version

from skfem import *
from skfem.io.meshio import from_meshio
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence

import numpy as np
from scipy.sparse import bmat, spmatrix
from scipy.sparse.linalg import LinearOperator, minres

import pygmsh


if version.parse(pygmsh.__version__) < version.parse('7.0.0'):
    class NullContextManager():
        def __enter__(self):
            return None
        def __exit__(self, *args):
            pass
    geometrycontext = NullContextManager()
else:
    geometrycontext = pygmsh.occ.Geometry()


try:
    try:
        from pyamgcl import amgcl  # v. 1.3.99+
    except ImportError:
        from pyamgcl import amg as amgcl
    from scipy.sparse.linalg import aslinearoperator

    def build_pc_amg(A: spmatrix, **kwargs) -> LinearOperator:
        """AMG preconditioner"""
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

    def mesh(self, geom=None, lcar=.1) -> MeshTet:
        with geometrycontext as g:
            if version.parse(pygmsh.__version__) < version.parse('7.0.0'):
                geom = pygmsh.opencascade.Geometry()
            else:
                geom = g
            geom.add_ellipsoid([0.]*3, self.semiaxes, lcar)
            if version.parse(pygmsh.__version__) < version.parse('7.0.0'):
                return from_meshio(
                    pygmsh.generate_mesh(geom))
            else:
                return from_meshio(
                    geom.generate_mesh())

    def pressure(self, x, y, z) -> np.ndarray:
        """Exact pressure at zero Grashof number.

        * McBain, G. D. (2016). `Creeping convection in a horizontally heated ellipsoid
        <http://people.eng.unimelb.edu.au/imarusic/proceedings/20/548/%20Paper.pdf>`_.
        *Proceedings of the Twentieth Australasian Fluid Mechanics Conference*, eq. 8

        """

        a, b, c = self.semiaxes
        return (a**2 * (3 * a**2 + b**2) * x * y
                / (3 * a**4 + 2 * a**2 * b**2 + 3 * b**4))

    def pressure_error(self):

        def form(v, w):
            return v * (w['p'] - self.pressure(*w.x))

        return LinearForm(form)


ellipsoid = Ellipsoid(.5, .3, .2)
mesh = ellipsoid.mesh(lcar=.1)

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

error_p = asm(ellipsoid.pressure_error(), basis['p'],
              p=basis['p'].interpolate(pressure))
l2error_p = np.sqrt(error_p.T @ Q @ error_p)

if __name__ == '__main__':

    from pathlib import Path

    print('L2 error in pressure:', l2error_p)

    mesh.save(Path(__file__).with_suffix('.vtk'),
              {'velocity': velocity[basis['u'].nodal_dofs].T,
               'pressure': pressure})
