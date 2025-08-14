# Prefer PDM SCM hook when available, but never hard-fail.
try:
    from pdm.backend.hooks.version import get_version_from_scm  # type: ignore
    _scm_ver = get_version_from_scm(".")
    __version__ = str(getattr(_scm_ver, "version", _scm_ver)) or "0+unknown"
except Exception:
    # Fall back to local resolver (installed dist, git describe, or default)
    try:
        from ._version import get_version_fallback
    except Exception:
        # ultra-conservative final fallback if even local import fails
        def get_version_fallback(*args, **kwargs):
            return "0+unknown"
    __version__ = get_version_fallback("pysonde")
