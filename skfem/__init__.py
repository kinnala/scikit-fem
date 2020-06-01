"""Support for wildcard import."""

from skfem.mesh import *  # noqa
from skfem.assembly import *  # noqa
from skfem.mapping import *  # noqa
from skfem.element import *  # noqa
from skfem.utils import *  # noqa
from skfem.version import __version__  # noqa

from skfem.assembly import __all__ as all_assembly
from skfem.mesh import __all__ as all_mesh
from skfem.element import __all__ as all_element


__all__ = all_mesh + all_assembly + all_element + [
    'MappingAffine',  # noqa
    'MappingIsoparametric',  # noqa
    'MappingMortar',  # noqa
    'adaptive_theta',  # noqa
    'build_pc_ilu',  # noqa
    'build_pc_diag',  # noqa
    'condense',  # noqa
    'derivative',  # noqa
    'L2_projection',  # noqa
    'project',  # noqa
    'solve',  # noqa
    'solver_direct_scipy',  # noqa
    'solver_iter_pcg',  # noqa
    'solver_iter_krylov',  # noqa
]
