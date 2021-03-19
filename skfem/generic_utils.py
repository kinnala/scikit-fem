import numpy as np


class HashableNdArray(np.ndarray):
    """Immutable ndarray with hashing support.

    Intended for enabling caching in other functions as default Python caching
    requires hashability of function arguments.  Note that if HashableNdArray
    is used to wrap an existing ndarray, the HashableNdArray becomes a view of
    the underlying array. Therefore modifying the underlying array is likely to
    break things.

    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and\
           isinstance(args[0], np.ndarray):
            obj = np.asarray(args[0]).view(cls)
            obj._hash = None
            obj.flags.writeable = False
        else:
            obj = super(HashableNdArray, cls).__new__(cls, *args, **kwargs)
        obj.flags.writeable = False
        return obj

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.tobytes())
        return self._hash

    def __eq__(self, other):
        if isinstance(other, HashableNdArray):
            return (self.__array__() == other.__array__()).all()
