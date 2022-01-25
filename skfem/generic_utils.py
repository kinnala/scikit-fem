import numpy as np

from numpy import ndarray


def hash_args(*args):
    """Return a tuple of hashes, with numpy support."""
    return tuple(hash(arg.tobytes())
                 if isinstance(arg, ndarray)
                 else hash(arg) for arg in args)


class OrientedFacetArray(ndarray):
    """An array of facet indices with orientation."""

    def __new__(cls, indices, ori):
        obj = np.asarray(indices).view(cls)
        obj.ori = np.array(ori, dtype=int)
        assert len(obj) == len(obj.ori)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.ori = getattr(obj, 'ori', None)

    @classmethod
    def by_subdomain(cls, m, ix):
        facets, counts = np.unique(m.t2f[:, ix], return_counts=True)
        facets = facets[counts == 1]
        ori = np.nonzero(np.isin(m.f2t[:, facets], ix).T)[1]
        return cls(facets, ori)
