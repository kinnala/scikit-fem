# mypy: ignore-errors
legacy = False
try:
    # Python 3.8+
    from importlib import metadata
except ImportError:
    try:
        import importlib_metadata as metadata_legacy
        legacy = True
    except ImportError:
        __version__ = "unknown"

try:
    if legacy:
        __version__ = metadata_legacy.version("scikit-fem")
    else:
        __version__ = metadata.version("scikit-fem")
except Exception:
    __version__ = "unknown"
