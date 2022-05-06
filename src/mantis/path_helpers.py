"""Contains helper functions for directory/path management/creation."""

from __future__ import annotations

import inspect
import os
import platform
import subprocess

from pathlib import Path


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
            #dir_ = Path.home() / f".{dirname}"  # e.g. /home/$USER/.tmap-detector
            dir_ = Path(f"/home/example-user/.{dirname}")
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


def open_directory_with_filebrowser(dir_to_open: Path) -> None:
    """
    Opens/runs filebrowser with default filebrowser application at the specified directory location.

    :param dir_to_open: path to directory.

    :raises ValueError: if passed a filepath, to prevent easily running malicious programs.
    """
    if not dir_to_open.is_dir():
        raise ValueError(
            f"Can only open a directory with this {inspect.currentframe().f_code.co_name}; "
            f"provided a path: {str(dir_to_open)!r}."
        )
    match platform.system():
        case "Linux":
            subprocess.Popen(["xdg-open", str(Path)])
        case "Windows":
            os.startfile(dir_to_open)  # type: ignore
        case "Darwin":
            subprocess.Popen(["open", str(Path)])
        case "Java":
            raise NotImplementedError("Java platform not (yet?) supported.")
        case _:
            raise NotImplementedError(
                f"Unrecognized/unsupported platform/OS: {platform.system()!r}."
            )


def get_datadir(pkg_name: str) -> Path:
    """Gets directory to store dataset(s) in local application directory."""
    return get_appdir(dirname=pkg_name) / "data"
