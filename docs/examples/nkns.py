from ns import *

from scipy.sparse.linalg import splu, spilu, LinearOperator, gmres


class KrylovBackwardFacingStep(BackwardFacingStep):

    def __init__(self,
                 length: float = 35.,
                 lcar: float = 1.):
        super().__init__(length, lcar)
        self.lu0 = splu(condense(self.S, I=self.I).T)
    
    def creeping(self):
        """return the solution for zero Reynolds number"""
        uvp = self.make_vector()
        uvp[self.I] = self.lu0.solve(
            condense(self.S, np.zeros_like(uvp), uvp, self.I)[1])
        return uvp


    def jacobian_solver(self,
                        uvp: np.ndarray,
                        reynolds: float,
                        rhs: np.ndarray) -> np.ndarray:
        duvp = self.make_vector() - uvp
        u = self.basis['u'].interpolate(self.split(uvp)[0])
        A = (self.S
             + reynolds * block_diag([asm(acceleration_jacobian,
                                          self.basis['u'], w=u),
                                      csr_matrix((self.basis['p'].N,)*2)]))
        A1 = condense(A, I=self.I)
        ilu = spilu(A1, 1e-5 * 1., 1e1 * 1.)
        _, rhs1 = condense(A, rhs, duvp, I=self.I)
        duvp[self.I], info = gmres(A1, rhs1, self.lu0.solve(rhs1), 1e-12,
                                   M=LinearOperator(ilu.L.shape, ilu.solve))

        if info:
            raise RuntimeError(info)
        else:
            return duvp


bfs = KrylovBackwardFacingStep(lcar=.2)

re = []
ax = bfs.mesh_plot()

def callback(k, reynolds, uvp):
    print(f'Re = {reynolds}')
    re.append(reynolds)

    ax = bfs.streamlines(bfs.streamfunction(bfs.split(uvp)[0]))
    ax.set_title(f'Re = {reynolds}')
    ax.get_figure().savefig(f'{name}-{reynolds:.2f}-psi.png',
                            bbox_inches="tight", pad_inches=0)
    
    if reynolds > 750.:
        raise RangeException


if __name__ == '__main__':

    from functools import partial
    from os.path import splitext
    from sys import argv

    from matplotlib.tri import Triangulation

    name = splitext(argv[0])[0]

    uvp0 = bfs.creeping()
                                
    try:
        natural(bfs, uvp0, 0., callback,
                lambda_stepsize0=50.,
                lambda_stepsize_max=150.,
                newton_tol=1e-9)
    except RangeException:
        print(f'Reynolds number sweep complete: {re}.')
