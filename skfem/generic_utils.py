import logging

from numpy import ndarray


def hash_args(*args):
    """Return a tuple of hashes, with numpy support."""
    return tuple(hash(arg.tobytes())
                 if isinstance(arg, ndarray)
                 else hash(arg) for arg in args)


class Log:
    """For debug messages that would otherwise sacrifice performance.

    To enable logging, call :meth:`Log.enable` or set the built-in ``logging``
    module level to ``logging.DEBUG``.

    """

    def __init__(self):
        self.logger = logging.getLogger("skfem")

    def __call__(self, msg):
        self.logger.debug(msg)

    @property
    def enabled(self):
        return self.logger.isEnabledFor(logging.DEBUG)

    def enable(self):
        logging.basicConfig(level=logging.DEBUG)


# imported in skfem.__init__
log = Log()
