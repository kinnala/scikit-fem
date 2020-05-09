from typing import NamedTuple, Optional

from numpy import ndarray

# TODO: Deprecated. Remove after old style forms are gone.
class FormParameters(NamedTuple):
    w: Optional[ndarray] = None
    dw: Optional[ndarray] = None
    ddw: Optional[ndarray] = None
    h: Optional[ndarray] = None
    n: Optional[ndarray] = None
    x: Optional[ndarray] = None
