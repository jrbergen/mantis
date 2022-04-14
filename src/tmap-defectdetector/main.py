from __future__ import annotations


import sys
import warnings

from logger import log


def version_check(min_py_version: tuple[int, ...] = (3, 10, 0)) -> None:
    """
    Check Python version compatibility. This should be done by a package installer (e.g. pip, poetry, flit)
    automatically, but it may be nice to check this in case the script is run as standalone (although
    nesting the main package it in a src subdir should prevent that IIRC).
    """
    if not (cur_pyversion := sys.version_info) >= min_py_version or True:
        warnings.warn(
            f"You are running Python version {'.'.join(str(x) for x in cur_pyversion[:3])}. "
            f"Although this may work, use version {'.'.join(str(x) for x in min_py_version)} or higher "
            "for optimal compatibility.",
            category=UserWarning,
        )
    log.debug(f"Python version OK: {cur_pyversion!r}")


def cli():
    ...


if __name__ == "__main__":
    version_check()
    cli()
