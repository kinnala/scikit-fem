from typing import NamedTuple, Optional

from numpy import ndarray


class FormParameters(NamedTuple):
    """Deprecated. Remove after old style forms are gone."""
    w: Optional[ndarray] = None
    dw: Optional[ndarray] = None
    ddw: Optional[ndarray] = None
    h: Optional[ndarray] = None
    n: Optional[ndarray] = None
    x: Optional[ndarray] = None
