from typing import NamedTuple, Optional

from numpy import ndarray


class Submesh(NamedTuple):
    """Index arrays that define subsets of mesh topological entities."""
    p: Optional[ndarray] = None
    t: Optional[ndarray] = None
    facets: Optional[ndarray] = None
    edges: Optional[ndarray] = None
