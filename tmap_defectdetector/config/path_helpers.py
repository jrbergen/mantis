"""Contains helper functions for directory management/creation."""

from __future__ import annotations

import os
import platform
import shutil
import site

from pathlib import Path
from typing import Optional


def get_appdir(dirname: str, make_if_non_exsiting: bool = True) -> Path:
    """
    Returns application directory in a cross-platform way.

    :param dirname: (str) directory name (e.g. Python package name).
    :param make_if_non_exsiting: (optional bool) creates the directory and any parent directories which don't exist yet,
        given the proper permissions are available. (Defaults to True)
    :returns: pathlib.Path object
    """
    match platform.system():
        case "Linux":
            dir_ = Path.home() / f".{dirname}"  # e.g. /home/$USER/.tmap-detector
        case "Windows":
            if (appdata_dir := os.getenv("LOCALAPPDATA")) is None:
                raise EnvironmentError("LOCALAPPDATA environment variable does not seem to be set.")
            dir_ = Path(
                str(appdata_dir), dirname
            )  # e.g. C:/Users/$USER/AppData/Local/tmap-detector
        case "Darwin":
            dir_ = (
                Path.home() / "Library" / "Preferences" / dirname
            )  # e.g.  ~/Library/Preferences/tmap-detector
        case "Java":
            raise NotImplementedError("Java platform not (yet?) supported.")
        case _:
            raise NotImplementedError(
                f"Unrecognized/unsupported platform/OS: {platform.system()!r}."
            )

    if make_if_non_exsiting and not dir_.exists():
        try:
            dir_.mkdir(parents=True)
        except PermissionError as e:
            raise PermissionError(
                f"Couldn't create application data directory for {dirname!r}: no write permission."
            ) from e

    return dir_.resolve(strict=True)


def get_datadir(pkg_name: str) -> Path:
    """Gets directory to store dataset(s) in local application directory."""
    return get_appdir(dirname=pkg_name) / "data"
