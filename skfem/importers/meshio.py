import warnings

warnings.warn("skfem.importers is removed in the next release. "
              "Use skfem.io instead.", DeprecationWarning)

from ..io.meshio import *
