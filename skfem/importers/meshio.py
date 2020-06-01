import warnings

from ..io.meshio import *  # noqa


warnings.warn("skfem.importers is removed in the next release. "
              "Use skfem.io instead.", DeprecationWarning)
