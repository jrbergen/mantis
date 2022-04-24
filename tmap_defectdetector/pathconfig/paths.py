"""
Contains application-wide runtime-initialized directory paths
and some functions to deal w/ cross-platform compatibility.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from tmap_defectdetector.pathconfig.path_helpers import get_appdir

_PKG_NAME: str = "tmapdd"

# Directories
DIR_APP: Path = get_appdir(dirname=_PKG_NAME)
"""Application directory"""

DIR_TMP: Path = Path(tempfile.gettempdir()) / _PKG_NAME
"""Temporary directory for per-session storage"""

FILE_LOG: Path = DIR_APP / f"log-{_PKG_NAME}.log"
"""Log file location"""

DIR_DATASETS: Path = DIR_APP / "datasets"
"""Directory for defect detection datasets and accompanying files/scripts."""
