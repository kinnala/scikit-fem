"""Support for wildcard import."""


def mark_unstable(fun):
    def call(*args, **kwargs):
        import warnings
        warnings.warn("You are using an unstable feature '{}' that might "
                      "change in the near future. It is not part of the "
                      "publicly documented API.".format(fun.__module__))
        return fun(*args, **kwargs)
    return call


from skfem.mesh import *
from skfem.assembly import *
from skfem.mapping import *
from skfem.element import *
from skfem.utils import *


__all__ = ['Mesh',
           'Mesh2D',
           'Mesh3D',
           'MeshHex',
           'MeshLine',
           'MeshQuad',
           'MeshTet',
           'MeshTri',
           'MeshMortar',
           'FacetBasis',
           'Basis',
           'InteriorBasis',
           'MortarBasis',
           'asm',
           'MappingAffine',
           'MappingIsoparametric',
           'adaptive_theta',
           'bilinear_form',
           'build_pc_ilu',
           'build_pc_diag',
           'condense',
           'derivative',
           'L2_projection',
           'linear_form',
           'functional',
           'rcm',
           'solve',
           'solver_direct_scipy',
           'solver_direct_umfpack',
           'solver_iter_pcg',
           'solver_iter_krylov',
           'Element',
           'ElementTriArgyris',
           'ElementH1',
           'ElementH2',
           'ElementHcurl',
           'ElementHdiv',
           'ElementHex1',
           'ElementTriMorley',
           'ElementQuad0',
           'ElementQuad1',
           'ElementQuad2',
           'ElementTetN0',
           'ElementTetP0',
           'ElementTetP1',
           'ElementTetP2',
           'ElementTetRT0',
           'ElementTriDG',
           'ElementTriP0',
           'ElementTriP1',
           'ElementTriP2',
           'ElementTriRT0',
           'ElementVectorH1',
           'ElementLineP1',
           'ElementLineP2']
