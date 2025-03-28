# flake8: noqa
import logging
from skfem.autodiff import *


logger = logging.getLogger(__name__)
logger.warning("Warning: skfem.experimental.autodiff "
               "has been moved to skfem.autodiff.")
