"""
Contains application-wide runtime-initialized directory paths
and some functions to deal w/ cross-platform compatibility.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import platform

_PKG_NAME: str = "defectdetector"


def get_appdir(dirname: str, make_if_non_exsiting: bool = True) -> Path:
    """
    Returns application directory in a cross-platform way.

    :param dirname: (str) directory name.
    :param make_if_non_exsiting: (optional bool) creates the directory and any parent directories which don't exist yet,
        given the proper permissions are available. (Defaults to True)
    :returns: pathlib.Path object
    """
    match platform.system():
        case "Linux":
            dir_ = Path.home() / f'.{dirname}'  # e.g. /home/$USER/.tmap-detector
        case "Windows":
            dir_ = Path(os.getenv("LOCALAPPDATA"), dirname)  # e.g. C:/Users/$USER/AppData/Local/tmap-detector
        case "Darwin":
            dir_ = Path.home() / 'Library' / 'Preferences' / dirname  # e.g.  ~/Library/Preferences/tmap-detector
        case "Java":
            raise NotImplementedError("Java platform not (yet?) supported.")
        case _:
            raise NotImplementedError(f"Unrecognized/unsupported platform/OS: {platform.system()!r}.")

    if make_if_non_exsiting and not dir_.exists():
        try:
            dir_.mkdir(parents=True)
        except PermissionError as e:
            raise PermissionError(f"Couldn't create application data directory for {_PKG_NAME!r}: no write permission.") from e

    return dir_.resolve(strict=True)

# Directories
DIR_APP: Path = get_appdir(_PKG_NAME)
"""Application directory"""

DIR_TMP: Path = Path(tempfile.gettempdir()) / _PKG_NAME
"""Temporary directory for per-session storage"""

FILE_LOG: Path = DIR_APP / f"log-{_PKG_NAME}.log"
"""Log file location"""


