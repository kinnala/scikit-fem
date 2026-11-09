"""Visualization of meshes and finite element solutions.

The backends are submodules: :mod:`skfem.visuals.matplotlib` (default),
:mod:`skfem.visuals.svg`, :mod:`skfem.visuals.web` and
:mod:`skfem.visuals.vedo`.  They all consume :class:`~skfem.visuals.data.
VisData` created by :func:`~skfem.visuals.data.sample`.

"""

import importlib


_DEFAULT_BACKEND = 'matplotlib'


def set_default_backend(name: str) -> None:
    """Set the backend used by ``Mesh.draw``, ``Basis.plot``, etc.

    Parameters
    ----------
    name
        Name of a submodule of :mod:`skfem.visuals`, e.g. ``'matplotlib'``
        or ``'web'``.

    """
    global _DEFAULT_BACKEND
    get_backend(name)
    _DEFAULT_BACKEND = name


def get_default_backend() -> str:
    """Return the name of the default backend."""
    return _DEFAULT_BACKEND


def get_backend(name=None):
    """Import and return the backend module."""
    if name is None:
        name = _DEFAULT_BACKEND
    if not isinstance(name, str):
        raise TypeError("The backend must be given as a string, e.g. "
                        "'matplotlib' or 'web'; got {}.".format(type(name)))
    return importlib.import_module('skfem.visuals.{}'.format(name))
