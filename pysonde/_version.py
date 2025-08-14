from __future__ import annotations
from typing import Optional

def get_version_fallback(dist_name: str = "pysonde") -> str:
    """
    Best-effort version resolver:
      1) importlib.metadata for installed dists
      2) `git describe --tags --dirty --always` if in a Git repo
      3) '0+unknown' as final fallback
    """
    # 1) Installed package?
    try:
        from importlib.metadata import version, PackageNotFoundError  # py3.8+
        try:
            return version(dist_name)
        except PackageNotFoundError:
            pass
    except Exception:
        pass

    # 2) Git describe?
    try:
        import subprocess, os
        # Resolve to repo root if module is imported from source tree
        here = os.path.abspath(os.path.dirname(__file__))
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty", "--always"],
            cwd=here,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            ver = result.stdout.strip()
            if ver:
                # Normalize PEP 440-ish (optional)
                return ver.replace("-", "+", 1).replace("-", ".")
    except Exception:
        pass

    # 3) Last resort
    return "0+unknown"
