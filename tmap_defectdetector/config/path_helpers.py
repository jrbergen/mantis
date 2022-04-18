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
            raise PermissionError(f"Couldn't create application data directory for {dirname!r}: no write permission.") from e

    return dir_.resolve(strict=True)

def get_datadir(pkg_name: str) -> Path:
    """Gets directory to store dataset(s) in local application directory."""
    return get_appdir(dirname=pkg_name) / 'data'


def add_default_dataset_urls_yaml_file(pkg_name: Optional[str] = None,
                                       dataset_urls_path_tgt: Optional[Path] = None) -> None:
    """
    Adds default dataset URL entries  from file included with package to dataset URLs specified (by user) in
    application if the entries don't exist yet.

    :param dataset_urls_path_tgt: Path to copy list of dataset URLs to if it doesn't exist.
    :param pkg_name: name of package used to find

    """

    from tmap_defectdetector.logger import log  # Hopefully this is the only file where avoiding circular imports this way is necessary.
    if None in (pkg_name, dataset_urls_path_tgt):
        from tmap_defectdetector.config.paths import DEFAULT_DATASET_URLS_FILE, _PKG_NAME
        pkg_name = _PKG_NAME
        dataset_urls_path_tgt = DEFAULT_DATASET_URLS_FILE

    if dataset_urls_path_tgt.exists():
        log.debug("File containing dataset URLs already exists; will not copy default from site-packages.")

    try:
        DATASET_URL_PACKAGE_PATH = Path(site.getsitepackages()[0]) / pkg_name / 'data' / 'dataset-urls.yaml'
    except IndexError:
        log.error("Couldn't find site-packages directory; "
                  f"will not copy file containing default dataset URLs from site-packages/{pkg_name}.")
        return

    if not DATASET_URL_PACKAGE_PATH.exists():
        log.warn("Couldn't find file containing dataset URLs in package directory; "
                 "will not copy default from site-packages.")
    else:
        if not dataset_urls_path_tgt.parent.exists:
            dataset_urls_path_tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATASET_URL_PACKAGE_PATH, dataset_urls_path_tgt)
        log.debug(f"No dataset URL file found; copied default dataset URL file {str(DATASET_URL_PACKAGE_PATH)} "
                  f"from package to {str(dataset_urls_path_tgt)}.")
