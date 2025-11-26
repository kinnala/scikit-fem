"""Support for wildcard import."""

from cudaskfem.mesh import *  # noqa
from cudaskfem.assembly import *  # noqa
from cudaskfem.mapping import *  # noqa
from cudaskfem.element import *  # noqa
from cudaskfem.utils import *  # noqa

from cudaskfem.assembly import __all__ as all_assembly
from cudaskfem.mesh import __all__ as all_mesh
from cudaskfem.element import __all__ as all_element

from .__about__ import __version__


__all__ = all_mesh + all_assembly + all_element + [  # noqa
    'MappingAffine',
    'MappingIsoparametric',
    'adaptive_theta',
    'build_pc_ilu',
    'build_pc_diag',
    'condense',
    'enforce',
    'penalize',
    'project',  # TODO remove due to deprecation
    'projection',  # TODO remove due to deprecation
    'solve',
    'bmat',
    'mpc',
    'solver_direct_scipy',
    'solver_eigen_scipy',
    'solver_eigen_scipy_sym',
    'solver_iter_pcg',
    'solver_iter_krylov',
    'solver_iter_cg',
    '__version__',
]
