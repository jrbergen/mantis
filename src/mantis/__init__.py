"""
Contains application-wide runtime-initialized names and directory paths.
"""
from __future__ import annotations

import platform
import tempfile
from pathlib import Path

from mantis.path_helpers import get_appdir
import importlib.metadata

APP_NAME: str = "mantis"

__version__ = importlib.metadata.version(APP_NAME)

# Directories
DIR_APP: Path = get_appdir(dirname=APP_NAME)
"""Application directory"""

DIR_TMP: Path = Path(tempfile.gettempdir()) / APP_NAME
"""Temporary directory for per-session storage"""

FILE_LOG: Path = DIR_APP / f"log-{APP_NAME}.log"
"""Log file location"""

DIR_DATASETS: Path = DIR_APP / "datasets"
"""Directory for defect detection datasets and accompanying files/scripts."""

TEXTUAL_LOGPATH: Path = DIR_APP / "textual.log"
"""Separate log file for TUI (which uses textual)"""


DEFAULT_ELPV_MODELPATH = (
    Path(DIR_TMP, "last_model_elpv.tflowmodel")
    if platform.system() == "Linux"
    else Path(DIR_APP, "last_model_elpv.tflowmodel")
)
"""Path to save ELPV CNN training weights to."""
