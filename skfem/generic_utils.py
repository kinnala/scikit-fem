from numpy import ndarray


def hash_args(*args):
    """Return a tuple of hashes, with numpy support."""
    return tuple(hash(arg.tobytes())
                 if isinstance(arg, ndarray)
                 else hash(arg) for arg in args)
