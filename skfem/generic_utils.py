import numpy as np

from numpy import ndarray


def hash_args(*args):
    """Return a tuple of hashes, with numpy support."""
    return tuple(hash(arg.tobytes())
                 if isinstance(arg, ndarray)
                 else hash(arg) for arg in args)


class OrientedBoundary(ndarray):
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


_REMOVAL_REGISTRY = []


class Removed:

    __slots__ = ("_version", "_message", "_era", "_qualname", "__doc__")

    def __init__(self, qualname=None, *, version, message, era=None):
        self._qualname = qualname
        self._version = version
        self._message = message
        self._era = era
        self.__doc__ = f"Removed in {version}. {message}"
        _REMOVAL_REGISTRY.append(self)

    def __set_name__(self, owner, name):
        # fires only for assignment inside a class body
        if self._qualname is None:
            self._qualname = f"{owner.__qualname__}.{name}"

    def _die(self, *args, **kwargs):
        from skfem import __version__          # lazy: avoids a circular import
        msg = (f"{self._qualname} was removed in scikit-fem {self._version} "
               f"(you are running {__version__}).\n{self._message}")
        if self._era is not None:
            msg += (f"\nThis is {self._era} API")
        raise AttributeError(msg)

    def __get__(self, obj, objtype=None):
        # class-level access must stay quiet for autodoc and getmembers
        if obj is None:
            return self
        self._die()

    __call__ = _die

    def __repr__(self):
        return f"<removed in {self._version}: {self._qualname}>"

    def as_dict(self):
        return {"name": self._qualname, "version": self._version,
                "message": self._message, "era": self._era}
